import torch

from models.mymodel import MyModel


class MyModelCFOnly(MyModel):
    def forward(self, user_id, item_id, user_review, item_review, return_dict=False):
        user_cf, item_cf = self.graph_encoder(user_id, item_id)
        cf_features = self._cf_features(user_cf, item_cf)

        cf_pred = self.cf_predict_layer(cf_features).squeeze(-1)
        cf_pred = (
            cf_pred
            + self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )

        if return_dict:
            return {"rating_pred": cf_pred, "cf_pred": cf_pred}
        return cf_pred

    def calculate_loss(self, user_id, item_id, user_review, item_review, rating):
        pred = self.forward(user_id, item_id, user_review, item_review)
        return self.loss_fn(pred, rating)
