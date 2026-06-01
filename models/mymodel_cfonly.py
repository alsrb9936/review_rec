import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from models.base_model import BaseModel
from models.mymodel import MyModel
import numpy as np

class MyModelCFOnly(MyModel):
    """Dual-side selective alignment with dot-product review scoring.

    Structure:
      user_review -> user_text -> user_shared / user_residual
      item_review -> item_text -> item_shared / item_residual
      review_signal = <user_shared, item_shared> / sqrt(d_text)

    Residual components are separated from the CF-aligned shared components
    but are not subtracted from the rating score. They are returned for
    diagnostics and optional anti-collapse regularization.
    """


    def forward(self, user_id, item_id, user_review, item_review, return_dict=False):
        user_cf, item_cf = self.graph_encoder(user_id, item_id)

        cf_features = self._cf_features(user_cf, item_cf)
        c_ui = self.cf_pair_layer(cf_features)
        cf_signal = self.cf_predict_layer(c_ui).squeeze(-1)


        user_text = self.user_review_encoder(user_review)
        item_text = self.item_review_encoder(item_review)


        # Dot-product review score. Residual is intentionally not subtracted.
        review_signal = self._dot_score(user_text, item_text)

        # Pair representation for contrastive alignment. This has the same dimensionality as c_ui.
        # z_shared_pair = user_shared * item_shared

        bias = (
            self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )

        alpha = torch.sigmoid(self.alpha_logit)
        rating_pred = alpha * cf_signal + (1.0 - alpha) * review_signal + bias

        if return_dict:
            return {
                "rating_pred": rating_pred,
                "cf_signal": cf_signal,
                "review_signal": review_signal,
                "bias": bias,
                "alpha": alpha.detach().expand_as(rating_pred),

                "user_cf": user_cf,
                "item_cf": item_cf,
                "c_ui": c_ui,

                "user_text": user_text,
                "item_text": item_text,
            }

        return rating_pred

    def calculate_loss(self, user_id, item_id, user_review, item_review, rating):
        outputs = self.forward(
            user_id=user_id,
            item_id=item_id,
            user_review=user_review,
            item_review=item_review,
            return_dict=True,
        )

        rating_loss = self.loss_fn(outputs["rating_pred"], rating)


        return (
            rating_loss
        )
