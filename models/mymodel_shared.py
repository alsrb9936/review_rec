import torch

from models.mymodel import MyModel


class MyModelShared(MyModel):
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
        z_shared, z_residual, shared_ratio, residual_ratio = self._orthogonal_decompose(
            z_review=z_review,
            c_ui=c_ui,
        )

        correction_features = torch.cat(
            [z_shared, c_ui, z_shared * c_ui, torch.abs(z_shared - c_ui)],
            dim=-1,
        )
        review_delta = self.shared_correction_layer(correction_features).squeeze(-1)
        rating_pred = cf_pred + self.review_scale * review_delta

        if return_dict:
            return {
                "rating_pred": rating_pred,
                "cf_pred": cf_pred,
                "review_delta": review_delta,
                "gate": torch.ones_like(review_delta),
                "shared_ratio": shared_ratio.squeeze(-1),
                "residual_ratio": residual_ratio.squeeze(-1),
                "c_ui": c_ui,
                "z_review": z_review,
                "z_shared": z_shared,
                "z_residual": z_residual,
            }
        return rating_pred
