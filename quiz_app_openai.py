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
    lesson_text: str
    questions_count: int = Field(default=3, ge=1, le=10)
    title_hint: Optional[str] = None


class ContentItem(BaseModel):
    type: Literal["text", "code", "image"]
    value: Optional[str] = None
    lang: Optional[str] = None
    url: Optional[str] = None
    alt: Optional[str] = None

    @model_validator(mode="after")
    def validate_content(self):
        if self.type == "text" and not self.value:
            raise ValueError("Для content.type='text' нужно поле value")
        if self.type == "code":
            if not self.value:
                raise ValueError("Для content.type='code' нужно поле value")
            if not self.lang:
                self.lang = "text"
        if self.type == "image" and not self.url:
            raise ValueError("Для content.type='image' нужно поле url")
        return self


class OptionItem(BaseModel):
    id: int
    content: List[ContentItem]

    @model_validator(mode="after")
    def validate_option(self):
        if not self.content:
            raise ValueError("У option.content должен быть хотя бы 1 элемент")
        return self


class QuestionItem(BaseModel):
    type: Literal["single", "multiple", "ordering"]
    content: List[ContentItem]
    options: List[OptionItem]
    correct: List[int]
    score: int = Field(ge=1, le=5)
    explanation: str

    @model_validator(mode="after")
    def validate_question(self):
        if not self.content:
            raise ValueError("У question.content должен быть хотя бы 1 элемент")

        option_ids = [opt.id for opt in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("У options должны быть уникальные id")

        for cid in self.correct:
            if cid not in option_ids:
                raise ValueError(f"correct содержит id={cid}, которого нет в options")

        if self.type == "single":
            if len(self.options) != 4:
                raise ValueError("Для type='single' должно быть ровно 4 варианта")
            if len(self.correct) != 1:
                raise ValueError("Для type='single' correct должен содержать ровно 1 id")

        elif self.type == "multiple":
            if len(self.options) != 4:
                raise ValueError("Для type='multiple' должно быть ровно 4 варианта")
            if len(self.correct) < 2:
                raise ValueError("Для type='multiple' correct должен содержать минимум 2 id")

        elif self.type == "ordering":
            if len(self.correct) != len(self.options):
                raise ValueError("Для type='ordering' correct должен содержать порядок всех option id")

        return self


class QuestionsResponse(BaseModel):
    quiz_id: int
    questions: List[QuestionItem]


def safe_text_content(value: str) -> dict:
    return {
        "type": "text",
        "value": value,
        "lang": None,
        "url": None,
        "alt": None,
    }


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


def extract_text_from_option(option: dict) -> str:
    content = option.get("content", [])
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and item.get("value"):
            parts.append(str(item["value"]))
    return " ".join(parts).strip()


def normalize_option(option, fallback_id: int):
    if isinstance(option, dict):
        option_id = option.get("id", fallback_id)
        try:
            option_id = int(option_id)
        except Exception:
            option_id = fallback_id

        content = option.get("content")

        if isinstance(content, list) and content:
            repaired_content = []
            for item in content:
                if isinstance(item, dict):
                    repaired_content.append({
                        "type": item.get("type", "text"),
                        "value": item.get("value"),
                        "lang": item.get("lang"),
                        "url": item.get("url"),
                        "alt": item.get("alt"),
                    })
                elif isinstance(item, str):
                    repaired_content.append(safe_text_content(item))

            if repaired_content:
                return {"id": option_id, "content": repaired_content}

        if isinstance(option.get("value"), str):
            return {
                "id": option_id,
                "content": [safe_text_content(option["value"])]
            }

    if isinstance(option, str):
        return {
            "id": fallback_id,
            "content": [safe_text_content(option)]
        }

    return {
        "id": fallback_id,
        "content": [safe_text_content(str(option))]
    }


def infer_correct_from_explanation(q_type: str, repaired_options: list, explanation: str) -> list:
    explanation_text = normalize_text(explanation)
    if not explanation_text:
        return []

    matched_ids = []
    for opt in repaired_options:
        opt_text = normalize_text(extract_text_from_option(opt))
        if opt_text and opt_text in explanation_text:
            matched_ids.append(opt["id"])

    matched_ids = unique_keep_order(matched_ids)

    if q_type == "single":
        return matched_ids[:1]
    if q_type == "multiple":
        return matched_ids
    return []


def build_fallback_explanation(q_type: str, repaired_options: list, correct: list) -> str:
    option_map = {opt["id"]: opt for opt in repaired_options}

    if q_type == "single":
        texts = [extract_text_from_option(option_map[cid]) for cid in correct if cid in option_map]
        return f"Правильный ответ: {', '.join(texts)}." if texts else "Правильный ответ определён на основе учебного текста."

    if q_type == "multiple":
        texts = [extract_text_from_option(option_map[cid]) for cid in correct if cid in option_map]
        return f"Правильные варианты: {', '.join(texts)}." if texts else "Правильные варианты определены на основе учебного текста."

    ordered = [extract_text_from_option(option_map[cid]) for cid in correct if cid in option_map]
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

        score = 1

        content = q.get("content")
        if not isinstance(content, list) or not content:
            if isinstance(q.get("text"), str) and q.get("text").strip():
                content = [safe_text_content(q["text"].strip())]
            else:
                content = [safe_text_content(f"Вопрос {q_index + 1}")]
        else:
            repaired_content = []
            for item in content:
                if isinstance(item, dict):
                    repaired_content.append({
                        "type": item.get("type", "text"),
                        "value": item.get("value"),
                        "lang": item.get("lang"),
                        "url": item.get("url"),
                        "alt": item.get("alt"),
                    })
                elif isinstance(item, str) and item.strip():
                    repaired_content.append(safe_text_content(item.strip()))
            content = repaired_content or [safe_text_content(f"Вопрос {q_index + 1}")]

        raw_options = q.get("options", [])
        repaired_options = []

        if isinstance(raw_options, list):
            repaired_options = [normalize_option(opt, idx) for idx, opt in enumerate(raw_options)]

        if q_type in {"single", "multiple"}:
            while len(repaired_options) < 4:
                repaired_options.append({
                    "id": len(repaired_options),
                    "content": [safe_text_content(f"Вариант {len(repaired_options) + 1}")]
                })
            repaired_options = repaired_options[:4]

        repaired_options = [
            {
                "id": idx,
                "content": opt.get("content", [safe_text_content(f"Вариант {idx + 1}")])
            }
            for idx, opt in enumerate(repaired_options)
        ]

        option_map = {opt["id"]: opt for opt in repaired_options}

        correct = q.get("correct")
        if not isinstance(correct, list):
            correct = []

        normalized_correct = []
        for c in correct:
            try:
                c_int = int(c)
                if c_int in option_map:
                    normalized_correct.append(c_int)
            except Exception:
                pass

        correct = unique_keep_order(normalized_correct)

        if not correct and "correct_answer" in q:
            correct_answer = q.get("correct_answer")
            if isinstance(correct_answer, str) and correct_answer.strip():
                correct_answer_norm = normalize_text(correct_answer)
                matched_ids = []
                for opt in repaired_options:
                    opt_text = normalize_text(extract_text_from_option(opt))
                    if opt_text == correct_answer_norm:
                        matched_ids.append(opt["id"])
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
                correct = [repaired_options[0]["id"]] if repaired_options else [0]

        elif q_type == "multiple":
            if len(explanation_based_correct) >= 2:
                correct = explanation_based_correct
            elif len(correct) < 2:
                if len(repaired_options) >= 2:
                    correct = [repaired_options[0]["id"], repaired_options[1]["id"]]
                elif len(repaired_options) == 1:
                    correct = [repaired_options[0]["id"]]

        elif q_type == "ordering":
            if len(correct) != len(repaired_options):
                explanation_text = normalize_text(explanation)
                ordered_ids = []
                for opt in repaired_options:
                    opt_text = normalize_text(extract_text_from_option(opt))
                    pos = explanation_text.find(opt_text) if opt_text else -1
                    if pos >= 0:
                        ordered_ids.append((pos, opt["id"]))

                if len(ordered_ids) == len(repaired_options):
                    ordered_ids.sort(key=lambda x: x[0])
                    correct = [item[1] for item in ordered_ids]
                else:
                    correct = [opt["id"] for opt in repaired_options]

        if bad_explanation(explanation):
            explanation = build_fallback_explanation(q_type, repaired_options, correct)

        repaired_questions.append({
            "type": q_type,
            "content": content,
            "options": repaired_options,
            "correct": correct,
            "score": score,
            "explanation": explanation,
        })

    while len(repaired_questions) < questions_count:
        idx = len(repaired_questions) + 1
        repaired_questions.append({
            "type": "single",
            "content": [safe_text_content(f"Вопрос {idx}")],
            "options": [
                {"id": 0, "content": [safe_text_content("Вариант 1")]},
                {"id": 1, "content": [safe_text_content("Вариант 2")]},
                {"id": 2, "content": [safe_text_content("Вариант 3")]},
                {"id": 3, "content": [safe_text_content("Вариант 4")]},
            ],
            "correct": [0],
            "score": 1,
            "explanation": "Правильный ответ определён автоматически.",
        })

    return {
        "quiz_id": quiz_id,
        "questions": repaired_questions
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
Ты создаёшь учебные тесты по тексту.

Нужно сгенерировать ровно {questions_count} вопросов.

Разрешённые типы:
- "single"
- "multiple"
- "ordering"

Верни JSON строго в таком формате:
{{
  "questions": [
    {{
      "type": "single",
      "content": [
        {{ "type": "text", "value": "Текст вопроса" }}
      ],
      "options": [
        {{ "id": 0, "content": [{{ "type": "text", "value": "Вариант 1" }}] }},
        {{ "id": 1, "content": [{{ "type": "text", "value": "Вариант 2" }}] }},
        {{ "id": 2, "content": [{{ "type": "text", "value": "Вариант 3" }}] }},
        {{ "id": 3, "content": [{{ "type": "text", "value": "Вариант 4" }}] }}
      ],
      "correct": [0],
      "score": 1,
      "explanation": "Осмысленное объяснение"
    }}
  ]
}}

Правила:
1. Все вопросы должны быть только по учебному тексту.
2. Не придумывай факты, которых нет в тексте.
3. Не используй абсурдные или случайные варианты.
4. Не смешивай языки.
5. explanation должен быть осмысленным и не пустым.
6. Не используй шаблонные explanation вроде "Options", "Explanation", "Краткое объяснение".
7. correct должен точно соответствовать правильным option.id.
8. Если type="single", correct должен содержать ровно один id.
9. Если type="multiple", укажи все правильные варианты.
10. Если type="ordering", correct должен содержать правильный порядок option.id.
11. Не повторяй одинаковые вопросы.
12. Верни только JSON без markdown и комментариев.
13. Перед возвратом JSON обязательно проверь, что поле correct точно соответствует правильным option.id.
14. explanation должен объяснять именно те варианты, которые указаны в correct.
15. Если type="multiple", укажи все правильные варианты, а не только часть.
16. Если type="single", правильный ответ должен быть однозначным и явно присутствовать среди options.
17. Не возвращай шаблонные explanation вроде "Options", "Explanation", "Краткое объяснение".
18. Для type="multiple" explanation должен перечислять именно все правильные варианты и не включать неправильные.
19. Для type="ordering" в options должны быть только элементы, которые действительно участвуют в порядке.
20. Для type="ordering" не добавляй лишний вариант, который не входит в правильную последовательность.
21. Если нельзя составить качественный ordering-вопрос без лишнего варианта, лучше создай single или multiple.
22. Если explanation говорит, что вариант неправильный, не включай его в correct.

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
        return service.generate_questions(
            lesson_text=payload.lesson_text,
            questions_count=payload.questions_count,
            title_hint=payload.title_hint,
            quiz_id=quiz_id,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))