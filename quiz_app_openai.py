import ast
import os
import json
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
app = FastAPI(title="Oqumi Question Generator API (OpenAI)")


class GenerateQuestionsRequest(BaseModel):
    lesson_text: dict
    questions_count: int = Field(default=3, ge=1, le=10)
    title_hint: Optional[str] = None


class QuestionItem(BaseModel):
    type: Literal["single", "multiple", "ordering"]
    text: str
    options: List[str]
    correct: List[int]
    score: int = Field(default=1, ge=1, le=5)
    explanation: str

    @model_validator(mode="after")
    def validate_question(self):
        if not self.text or not self.text.strip():
            raise ValueError("У question.text должен быть текст")

        if not self.explanation or not self.explanation.strip():
            raise ValueError("У question.explanation должен быть текст")

        if not self.options:
            raise ValueError("У question.options должен быть массив вариантов")

        for option in self.options:
            if not isinstance(option, str) or not option.strip():
                raise ValueError("Каждый option должен быть непустой строкой")

        option_ids = list(range(len(self.options)))

        for cid in self.correct:
            if cid not in option_ids:
                raise ValueError(f"correct содержит индекс={cid}, которого нет в options")

        if self.type == "single":
            if len(self.options) != 4:
                raise ValueError("Для type='single' должно быть ровно 4 варианта")
            if len(self.correct) != 1:
                raise ValueError("Для type='single' correct должен содержать ровно 1 индекс")

        elif self.type == "multiple":
            if len(self.options) != 4:
                raise ValueError("Для type='multiple' должно быть ровно 4 варианта")
            if len(self.correct) < 2:
                raise ValueError("Для type='multiple' correct должен содержать минимум 2 индекса")

        elif self.type == "ordering":
            if len(self.options) < 2:
                raise ValueError("Для type='ordering' должно быть минимум 2 варианта")
            if len(self.correct) != len(self.options):
                raise ValueError("Для type='ordering' correct должен содержать порядок всех индексов options")

        return self


class QuestionsResponse(BaseModel):
    quiz_id: int
    questions: List[QuestionItem]


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split()).lower()


def unique_keep_order(values: List[int]) -> List[int]:
    seen = set()
    result = []
    for v in values:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def bad_explanation(text: str) -> bool:
    t = normalize_text(text)
    bad_values = {
        "",
        "options",
        "option",
        "краткое объяснение",
        "объяснение",
        "answer",
        "correct answer",
        "explanation",
    }
    return t in bad_values or len(t) < 12


def extract_text_from_stringified_object(value: str) -> str:
    """Если в text/value случайно пришла строка вида "{'id': 0, 'text': 'SELECT'}", берём только text."""
    if not isinstance(value, str):
        return ""

    cleaned = value.strip()
    if not cleaned:
        return ""

    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        return cleaned

    try:
        parsed = json.loads(cleaned)
    except Exception:
        try:
            parsed = ast.literal_eval(cleaned)
        except Exception:
            return cleaned

    if isinstance(parsed, dict):
        if isinstance(parsed.get("text"), str):
            return parsed["text"].strip()
        if isinstance(parsed.get("value"), str):
            return parsed["value"].strip()

    return cleaned


def extract_text_from_content(content) -> str:
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if isinstance(item, dict):
            value = item.get("text") or item.get("value") or ""
            value = extract_text_from_stringified_object(str(value))
            if value:
                parts.append(value)
        elif isinstance(item, str):
            value = extract_text_from_stringified_object(item)
            if value:
                parts.append(value)

    return " ".join(parts).strip()


def extract_question_text(question: dict, fallback_text: str) -> str:
    if isinstance(question.get("text"), str) and question["text"].strip():
        return extract_text_from_stringified_object(question["text"])

    content_text = extract_text_from_content(question.get("content"))
    if content_text:
        return content_text

    return fallback_text


def extract_text_from_option(option: dict) -> str:
    if not isinstance(option, dict):
        return ""

    if isinstance(option.get("text"), str) and option["text"].strip():
        return extract_text_from_stringified_object(option["text"])

    if isinstance(option.get("value"), str) and option["value"].strip():
        return extract_text_from_stringified_object(option["value"])

    content_text = extract_text_from_content(option.get("content"))
    if content_text:
        return content_text

    return ""


def normalize_option(option, fallback_id: int) -> dict:
    if isinstance(option, dict):
        option_id = option.get("id", fallback_id)
        try:
            option_id = int(option_id)
        except Exception:
            option_id = fallback_id

        option_text = extract_text_from_option(option)
        if option_text:
            return {"id": option_id, "value": option_text}

    elif isinstance(option, str):
        option_text = extract_text_from_stringified_object(option)
        if option_text:
            return {"id": fallback_id, "value": option_text}

    return {"id": fallback_id, "value": f"Вариант {fallback_id + 1}"}


def infer_correct_from_explanation(q_type: str, repaired_options: list, explanation: str) -> list:
    explanation_text = normalize_text(explanation)
    if not explanation_text:
        return []

    matched_ids = []
    for idx, opt_text in enumerate(repaired_options):
        opt_text_norm = normalize_text(opt_text)
        if opt_text_norm and opt_text_norm in explanation_text:
            matched_ids.append(idx)

    matched_ids = unique_keep_order(matched_ids)

    if q_type == "single":
        return matched_ids[:1]
    if q_type == "multiple":
        return matched_ids
    return []


def build_fallback_explanation(q_type: str, repaired_options: list, correct: list) -> str:
    if q_type == "single":
        texts = [repaired_options[cid] for cid in correct if 0 <= cid < len(repaired_options)]
        return f"Правильный ответ: {', '.join(texts)}." if texts else "Правильный ответ определён на основе учебного текста."

    if q_type == "multiple":
        texts = [repaired_options[cid] for cid in correct if 0 <= cid < len(repaired_options)]
        return f"Правильные варианты: {', '.join(texts)}." if texts else "Правильные варианты определены на основе учебного текста."

    ordered = [repaired_options[cid] for cid in correct if 0 <= cid < len(repaired_options)]
    return f"Правильный порядок: {' -> '.join(ordered)}." if ordered else "Правильный порядок определён на основе учебного текста."


def repair_generated_questions(data: dict, quiz_id: int, questions_count: int) -> dict:
    questions = data.get("questions", [])
    repaired_questions = []

    if not isinstance(questions, list):
        questions = []

    for q_index, q in enumerate(questions[:questions_count]):
        if not isinstance(q, dict):
            continue

        q_type = q.get("type", "single")
        if q_type not in {"single", "multiple", "ordering"}:
            q_type = "single"

        question_text = extract_question_text(q, fallback_text=f"Вопрос {q_index + 1}")
        score = 1

        raw_options = q.get("options", [])
        normalized_option_objects = []

        if isinstance(raw_options, list):
            normalized_option_objects = [normalize_option(opt, idx) for idx, opt in enumerate(raw_options)]

        if q_type in {"single", "multiple"}:
            while len(normalized_option_objects) < 4:
                normalized_option_objects.append({
                    "id": len(normalized_option_objects),
                    "value": f"Вариант {len(normalized_option_objects) + 1}",
                })
            normalized_option_objects = normalized_option_objects[:4]

        elif q_type == "ordering":
            while len(normalized_option_objects) < 2:
                normalized_option_objects.append({
                    "id": len(normalized_option_objects),
                    "value": f"Элемент {len(normalized_option_objects) + 1}",
                })

        old_id_to_new_id = {}
        repaired_options = []
        for new_id, opt in enumerate(normalized_option_objects):
            old_id_to_new_id[opt["id"]] = new_id
            repaired_options.append(
                opt.get("value") or opt.get("text") or f"Вариант {new_id + 1}"
            )

        option_ids = set(range(len(repaired_options)))

        correct = q.get("correct")
        if not isinstance(correct, list):
            correct = []

        normalized_correct = []
        for c in correct:
            try:
                c_int = int(c)
                if c_int in old_id_to_new_id:
                    normalized_correct.append(old_id_to_new_id[c_int])
                elif c_int in option_ids:
                    normalized_correct.append(c_int)
            except Exception:
                pass

        correct = unique_keep_order(normalized_correct)

        if not correct and "correct_answer" in q:
            correct_answer = q.get("correct_answer")
            if isinstance(correct_answer, str) and correct_answer.strip():
                correct_answer_norm = normalize_text(correct_answer)
                matched_ids = []
                for idx, opt_text in enumerate(repaired_options):
                    if normalize_text(opt_text) == correct_answer_norm:
                        matched_ids.append(idx)
                if matched_ids:
                    correct = matched_ids[:1]

        explanation = q.get("explanation", "")
        if not isinstance(explanation, str):
            explanation = str(explanation)
        explanation = explanation.strip()

        explanation_based_correct = infer_correct_from_explanation(
            q_type=q_type,
            repaired_options=repaired_options,
            explanation=explanation,
        )

        if q_type == "single":
            if len(explanation_based_correct) == 1:
                correct = explanation_based_correct
            elif len(correct) != 1:
                correct = [0] if repaired_options else [0]

        elif q_type == "multiple":
            if len(explanation_based_correct) >= 2:
                correct = explanation_based_correct
            elif len(correct) < 2:
                if len(repaired_options) >= 2:
                    correct = [0, 1]
                elif len(repaired_options) == 1:
                    correct = [0]

        elif q_type == "ordering":
            if len(correct) != len(repaired_options):
                explanation_text = normalize_text(explanation)
                ordered_ids = []
                for idx, opt_text in enumerate(repaired_options):
                    opt_text_norm = normalize_text(opt_text)
                    pos = explanation_text.find(opt_text_norm) if opt_text_norm else -1
                    if pos >= 0:
                        ordered_ids.append((pos, idx))

                if len(ordered_ids) == len(repaired_options):
                    ordered_ids.sort(key=lambda x: x[0])
                    correct = [item[1] for item in ordered_ids]
                else:
                    correct = list(range(len(repaired_options)))

        if bad_explanation(explanation):
            explanation = build_fallback_explanation(q_type, repaired_options, correct)

        repaired_questions.append({
            "type": q_type,
            "text": question_text,
            "options": repaired_options,
            "correct": correct,
            "score": score,
            "explanation": explanation,
        })

    while len(repaired_questions) < questions_count:
        idx = len(repaired_questions) + 1
        repaired_questions.append({
            "type": "single",
            "text": f"Вопрос {idx}",
            "options": [
                "Вариант 1",
                "Вариант 2",
                "Вариант 3",
                "Вариант 4",
            ],
            "correct": [0],
            "score": 1,
            "explanation": "Правильный ответ определён автоматически.",
        })

    return {
        "quiz_id": quiz_id,
        "questions": repaired_questions,
    }


class OpenAIQuestionService:
    def __init__(self, model: Optional[str] = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Не задан OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    def build_prompt(self, lesson_text: str, questions_count: int, title_hint: Optional[str]) -> str:
        title_hint_text = f"\nТема: {title_hint}\n" if title_hint else ""

        return f"""
Ты создаёшь учебные тесты по учебному тексту.

Нужно сгенерировать ровно {questions_count} вопросов.

Разрешённые типы вопросов:
- "single"
- "multiple"
- "ordering"

Верни JSON строго в таком формате:
{{
  "questions": [
    {{
      "type": "single",
      "text": "Текст вопроса",
      "options": [
        "Вариант 1",
        "Вариант 2",
        "Вариант 3",
        "Вариант 4"
      ],
      "correct": [1],
      "score": 1,
      "explanation": "Осмысленное объяснение правильного ответа"
    }}
  ]
}}

Правила:
1. Верни только JSON без markdown, без комментариев и без дополнительного текста.
2. Количество вопросов должно быть ровно {questions_count}.
3. Все вопросы должны быть только по учебному тексту.
4. Не придумывай факты, которых нет в учебном тексте.
5. Не используй абсурдные, случайные или слишком очевидно неправильные варианты.
6. Не смешивай языки.
7. Не используй поля content, value, lang, url, alt.
8. Не используй Tiptap-структуру.
9. Для текста вопроса используй только поле "text".
10. options должен быть массивом строк.
11. Не используй id, text или value внутри options.
12. У каждого single/multiple вопроса должно быть ровно 4 варианта ответа.
13. correct должен содержать индексы правильных вариантов из массива options.
14. Индексы в correct начинаются с 0.
15. score всегда должен быть 1.
16. explanation должен быть осмысленным и не пустым.
17. explanation должен объяснять именно правильный ответ.
18. Не используй шаблонные explanation вроде "Options", "Explanation", "Краткое объяснение".
19. Не повторяй одинаковые вопросы.

Правила для type="single":
20. correct должен содержать ровно один индекс.
21. Правильный ответ должен быть однозначным.
22. Правильный ответ должен явно присутствовать среди options.
23. Не ставь правильный ответ всегда на позицию 0.
24. Перемешивай варианты так, чтобы правильный ответ мог быть на позиции 0, 1, 2 или 3.

Правила для type="multiple":
25. correct должен содержать несколько индексов.
26. Укажи все правильные варианты, а не только часть.
27. explanation должен перечислять именно все правильные варианты.
28. explanation не должен включать неправильные варианты как правильные.
29. options должен содержать не только правильные, но и неправильные варианты.
30. correct не должен содержать все индексы [0, 1, 2, 3].
31. В multiple должно быть 2 или 3 правильных ответа, но не 4.
32. Минимум один вариант должен быть неправильным.

Правила для type="ordering":
33. correct должен содержать правильный порядок индексов options.
34. В options должны быть только элементы, которые действительно участвуют в порядке.
35. Не добавляй лишний вариант, который не входит в правильную последовательность.
36. Если нельзя составить качественный ordering-вопрос, лучше создай single или multiple.
37. Перемешивай options.
38. options НЕ должен быть сразу в правильном порядке.
39. correct должен показывать правильный порядок индексов после перемешивания.
40. Не возвращай correct всегда [0, 1, 2, 3].
Важно:
41. Пример JSON выше показывает только структуру.
42. Не копируй correct из примера.
43. Не возвращай варианты в виде объектов: {{"id": 0, "value": "..."}}.
44. Каждый элемент options должен быть простой строкой, например: "SELECT".

{title_hint_text}
Учебный текст:
{lesson_text}
""".strip()

    def generate_questions(
        self,
        lesson_text: str,
        questions_count: int = 3,
        title_hint: Optional[str] = None,
        quiz_id: int = 0,
    ) -> QuestionsResponse:
        prompt = self.build_prompt(lesson_text, questions_count, title_hint)

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            store=False,
        )

        raw_text = (response.output_text or "").strip()
        if not raw_text:
            raise ValueError("OpenAI вернула пустой ответ")

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"OpenAI вернула не-JSON:\n{raw_text}")
            parsed = json.loads(raw_text[start:end + 1])

        try:
            repaired = repair_generated_questions(
                parsed,
                quiz_id=quiz_id,
                questions_count=questions_count,
            )
            result = QuestionsResponse.model_validate(repaired)
        except ValidationError as e:
            raise ValueError(f"JSON после repair не прошёл валидацию: {e}\nRAW:\n{raw_text}")

        return result


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/quizzes/questions/generate/{quiz_id}", response_model=QuestionsResponse)
def generate_questions(quiz_id: int, payload: GenerateQuestionsRequest):
    try:
        service = OpenAIQuestionService()

        lesson_text = json.dumps(payload.lesson_text, ensure_ascii=False)

        return service.generate_questions(
            lesson_text=lesson_text,
            questions_count=payload.questions_count,
            title_hint=payload.title_hint,
            quiz_id=quiz_id,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
