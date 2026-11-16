"""Loss and counterfactual unit tests. They do not need a GPU or the datasets."""

from __future__ import annotations

import unittest

import torch

from src.counterfactual import excess_green_mask, make_boundary_counterfactuals, make_counterfactuals
from src.dgpldr import StyleSensitiveDistillation
from src.losses import (
    feature_consistency_loss,
    lesion_asymmetric_loss,
    logit_prototype_contrastive_loss,
    prediction_consistency_loss,
    prototype_stability_loss,
)
from src.prototypes import LogitPrototypeMemory


class LossTests(unittest.TestCase):
    def test_feature_consistency_is_zero_for_identical_embeddings(self):
        embedding = torch.randn(4, 16)
        loss = feature_consistency_loss(embedding, embedding)
        self.assertLess(float(loss), 1e-6)

    def test_feature_consistency_rises_when_views_differ(self):
        embedding = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
        other = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
        same = feature_consistency_loss(embedding, embedding)
        different = feature_consistency_loss(embedding, other)
        self.assertGreater(float(different), float(same))

    def test_prediction_consistency_is_zero_for_the_same_logits(self):
        logits = torch.randn(4, 5)
        loss = prediction_consistency_loss(logits, logits)
        self.assertLess(float(loss), 1e-5)

    def test_prediction_consistency_accepts_multiple_views(self):
        logits = torch.randn(3, 4)
        views = logits.unsqueeze(1).repeat(1, 2, 1)
        loss = prediction_consistency_loss(logits, views)
        self.assertLess(float(loss), 1e-5)

    def test_prototype_stability_is_zero_when_counterfactuals_match(self):
        embedding = torch.randn(6, 8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = prototype_stability_loss(embedding, embedding, labels, num_classes=3)
        self.assertLess(float(loss), 1e-6)

    def test_logit_prototype_loss_decreases_when_logits_match_their_class(self):
        prototypes = torch.eye(3)
        good_logits = prototypes[torch.tensor([0, 1, 2])] * 4
        bad_logits = prototypes[torch.tensor([1, 2, 0])] * 4
        labels = torch.tensor([0, 1, 2])
        good = logit_prototype_contrastive_loss(good_logits, labels, prototypes, temperature=0.8)
        bad = logit_prototype_contrastive_loss(bad_logits, labels, prototypes, temperature=0.8)
        self.assertLess(float(good), float(bad))


class MemoryTests(unittest.TestCase):
    def test_memory_initializes_then_moves_with_momentum(self):
        memory = LogitPrototypeMemory(num_classes=2, momentum=0.5)
        logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        labels = torch.tensor([0, 0])
        memory.update(logits, labels)
        self.assertTrue(bool(memory.initialized[0]))
        self.assertFalse(bool(memory.initialized[1]))
        memory.update(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))
        self.assertAlmostEqual(float(memory.prototypes[0, 0]), 0.5, places=5)


class CounterfactualTests(unittest.TestCase):
    def test_leaf_mask_covers_a_green_region(self):
        image = torch.zeros(1, 3, 32, 32)
        image[:, 1, 8:24, 8:24] = 0.8
        image[:, 0, 8:24, 8:24] = 0.1
        image[:, 2, 8:24, 8:24] = 0.1
        mask = excess_green_mask(image)
        self.assertGreater(float(mask[0, 0, 16, 16]), 0.5)
        self.assertLess(float(mask[0, 0, 0, 0]), 0.5)

    def test_counterfactual_shapes_and_leaf_preservation(self):
        image = torch.rand(2, 3, 40, 40)
        image[:, 1, 10:30, 10:30] = 0.85
        image[:, 0, 10:30, 10:30] = 0.15
        image[:, 2, 10:30, 10:30] = 0.15
        views, names = make_counterfactuals(image, num_views=3)
        self.assertEqual(tuple(views.shape), (2, 3, 3, 40, 40))
        self.assertEqual(names, ["background", "illumination", "style"])
        mask = excess_green_mask(image) > 0.5
        leaf_delta = (views[:, 0] - image).abs()[mask.expand_as(image)].mean()
        background = ~mask
        background_delta = (views[:, 0] - image).abs()[background.expand_as(image)].mean()
        self.assertLess(float(leaf_delta), float(background_delta))

    def test_boundary_edit_keeps_the_lesion_and_changes_the_background(self):
        image = torch.zeros(3, 96, 96)
        image[1, 16:80, 16:80] = 0.75
        image[0, 16:80, 16:80] = 0.12
        image[2, 16:80, 16:80] = 0.12
        image[0, 40:58, 40:58] = 0.62
        image[1, 40:58, 40:58] = 0.24
        image[2, 40:58, 40:58] = 0.08
        views, preserve, preserve_cf, _, background_cf = make_boundary_counterfactuals(image, num_views=1)
        self.assertEqual(tuple(views.shape), (1, 3, 96, 96))
        self.assertGreater(float(preserve.sum()), 10.0)
        lesion = image[:, 44:54, 44:54].mean()
        kept = views[0][:, preserve_cf[0, 0] > 0.5]
        self.assertGreater(kept.numel(), 0)
        self.assertLess(float((kept.mean() - lesion).abs()), 0.25)
        self.assertGreater(float((views[0] - image).abs()[:, background_cf[0, 0] > 0.5].mean()), 0.05)

    def test_asymmetric_loss_is_zero_when_the_edit_does_not_move_features(self):
        feat = torch.randn(2, 4, 8, 8)
        preserve = torch.zeros(2, 1, 8, 8)
        preserve[:, :, :, 4:] = 1
        background = torch.zeros(2, 1, 8, 8)
        background[:, :, :, :3] = 1
        keep, change = lesion_asymmetric_loss(
            feat,
            feat.unsqueeze(1),
            preserve,
            preserve.unsqueeze(1),
            background,
            background.unsqueeze(1),
        )
        self.assertEqual(float(keep), 0.0)
        self.assertEqual(float(change), 0.0)


class DistillationTests(unittest.TestCase):
    def test_ssd_preserves_feature_shape(self):
        module = StyleSensitiveDistillation()
        feat = torch.randn(2, 8, 6, 6)
        feat_aug = feat + 0.01 * torch.randn_like(feat)
        refined, refined_aug = module(feat, feat_aug)
        self.assertEqual(tuple(refined.shape), tuple(feat.shape))
        self.assertEqual(tuple(refined_aug.shape), tuple(feat_aug.shape))
        self.assertTrue(torch.isfinite(refined).all())


if __name__ == "__main__":
    unittest.main()
