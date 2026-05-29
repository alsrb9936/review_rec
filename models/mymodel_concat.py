import torch
import torch.nn as nn

from models.mymodel import MyModel


class MyModelConcat(MyModel):
    def __init__(self, cfg, norm_adj):
        super().__init__(cfg, norm_adj)
        self.concat_predict_layer = nn.Sequential(
            nn.Linear(self.d_text * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )
        self._init_linear_block(self.concat_predict_layer)

    def forward(self, user_id, item_id, user_review, item_review, return_dict=False):
        user_cf, item_cf = self.graph_encoder(user_id, item_id)
        cf_features = self._cf_features(user_cf, item_cf)
        c_ui = self.cf_pair_layer(cf_features)

        cf_pred = self.cf_predict_layer(cf_features).squeeze(-1)
        cf_pred = (
            cf_pred
            + self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )

        z_review = self._review_pair(user_review, item_review)
        review_delta = self.concat_predict_layer(torch.cat([c_ui, z_review], dim=-1)).squeeze(-1)
        rating_pred = cf_pred + self.review_scale * review_delta

        if return_dict:
            return {
                "rating_pred": rating_pred,
                "cf_pred": cf_pred,
                "review_delta": review_delta,
                "c_ui": c_ui,
                "z_review": z_review,
            }
        return rating_pred

    def calculate_loss(self, user_id, item_id, user_review, item_review, rating):
        pred = self.forward(user_id, item_id, user_review, item_review)
        return self.loss_fn(pred, rating)
