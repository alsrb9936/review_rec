import torch
from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig
from typing import Optional
from dataset.base_dataset import BaseDataset


class DeepCoNNDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DictConfig,
        word_dict: dict,
        split: str = "train",
    ):
        super().__init__()
        self.word_dict = word_dict
        self.review_length = int(cfg.data.review_length)
        self.review_count = int(cfg.data.review_count)
        self.pad_id = int(cfg.data.pad_id)
        self.lowest_r_count = int(cfg.data.lowest_review_count)  # 특정 user/item이 작성한 최소 리뷰 개수
        self.retain_rui = False  # train에서는 user와 item의 공통 리뷰를 유지, valid/test에서는 제거 
        if split == "train":
            self.retain_rui = bool(cfg.data.retain_rui) # 최종 샘플에서 user와 item의 공통 review를 유지할지 여부

        df["review_text"] = df["review_text"].apply(self._review2id) # 토큰화 후 숫자 ID로 변환
        self.sparse_idx = set()  # 희소한 샘플의 인덱스를 임시 저장하고, 마지막에 제거

        user_reviews = self._get_reviews(df)  # 각 user의 리뷰 리스트 수집
        item_reviews = self._get_reviews(df, 'user_id', 'item_id') # 각 item의 리뷰 리스트 수집
        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(df["item_id"].values, dtype=torch.long)
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)


        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.ratings = rating[[idx for idx in range(rating.shape[0]) if idx not in self.sparse_idx]]

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
            "user_reviews": self.user_reviews[idx],
            "item_reviews": self.item_reviews[idx],
        }

    def __len__(self):
        return self.ratings.shape[0]
    
    def _get_reviews(self, df, lead='user_id', costar='item_id'):
        # 각 학습 데이터에 대해 해당 사용자/아이템의 모든 리뷰를 모아서 생성
        reviews_by_lead = dict(list(df[[costar, 'review_text']].groupby(df[lead])))  # 각 user/item별 리뷰 모음
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # lead에 해당하는 모든 리뷰를 가져옴: DataFrame
            if self.retain_rui:
                reviews = df_data['review_text'].to_list()  # lead의 모든 리뷰를 가져옴: 리스트
            else:
                reviews = df_data['review_text'][df_data[costar] != costar_id].to_list()  # lead와 costar가 함께 등장한 현재 리뷰는 제외
            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.pad_id] * r_length] * (r_count - len(reviews))  # 리뷰 개수를 고정
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # 각 리뷰의 길이를 고정
        return reviews

    def _review2id(self, review):  # 하나의 리뷰 문자열을 단어 단위로 나누고 숫자 ID로 변환
        if not isinstance(review, str):
            return []  # pandas 관련 문제로 보이며, 빈 문자열로 읽힌 리뷰가 float 타입이 되는 경우가 있음
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # 단어를 숫자 ID로 매핑
            else:
                wids.append(self.pad_id)
        return wids
