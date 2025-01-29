from icecream import ic
from tensorflow.keras.datasets import cifar10
from typing import Final

from haio.common import haio_hash, img_to_url
from haio.types import QuestionConfig, Answer
from haio.worker_io.types import Worker_IO

from vec_to_img import vec_to_img


class VirtualHuman_IO(Worker_IO):
    labels = [
        "Airplane",
        "Car",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
    (x_train, y_train), _ = cifar10.load_data()
    answer_dict: Final[dict[str, Answer]] = {}
    for i, data in enumerate(x_train):
        answer_dict[haio_hash(img_to_url(vec_to_img(data), "image/png"))] = labels[
            y_train[i][0]
        ]

    def __init__(self) -> None:
        self.asked: dict[str, Answer] = {}

    def ask(self, question_config: QuestionConfig) -> str:
        question_config_hash = haio_hash(question_config)

        # GEMINI_IOでは、回答を取得せずに同じ質問を複数回聞くことはできない
        # 既に質問済みならエラーを返す
        if question_config_hash in self.asked:
            raise Exception("already asking")

        question_elem = next(
            (item for item in question_config["question"] if item["tag"] == "img"), None
        )
        if question_elem is None:
            raise Exception("question is invalid")

        img_url = question_elem["src"]

        print("VirtualHuman Question Config Hash:", question_config_hash)

        answer: Answer = self.answer_dict[haio_hash(img_url)]

        if answer:
            self.asked[question_config_hash] = answer
        else:
            raise Exception("The model returned empty response.")

        return question_config_hash

    def is_finished(self, id: str) -> bool:
        # idはquestion_config_hash
        if id not in self.asked:
            raise Exception("never asked")
        return self.asked[id] != ""  # 実質的には常にTrue

    def get_answer(self, id: str) -> Answer:
        # idはquestion_config_hash
        if id not in self.asked:
            raise Exception("never asked")
        tmp = self.asked[id]
        self.asked[id] = ""
        self.asked.pop(id)
        return tmp

    async def ask_get_answer(self, question_config: QuestionConfig) -> Answer:
        id = self.ask(question_config=question_config)
        return self.get_answer(id)
