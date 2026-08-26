import unittest
from types import SimpleNamespace

import torch

from models.TFPS import Model


def make_config(**overrides):
    values = dict(
        task_name='long_term_forecast',
        seq_len=96,
        pred_len=96,
        enc_in=7,
        c_out=1,
        patch_len=16,
        stride=8,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        factor=1,
        dropout=0.0,
        activation='gelu',
        tfps_t_num_experts=4,
        tfps_t_top_k=2,
        tfps_f_num_experts=4,
        tfps_f_top_k=2,
        tfps_subspace_dim=0,
        tfps_expert_hidden=32,
        tfps_eta=5.0,
        tfps_beta=0.1,
        tfps_time_loss_weight=1.0,
        tfps_frequency_loss_weight=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TFPSTest(unittest.TestCase):
    def test_forward_backward_and_routing_shapes(self):
        model = Model(make_config())
        x_enc = torch.randn(2, 96, 7)
        output = model(x_enc, None, None, None)

        self.assertEqual(tuple(output.shape), (2, 96, 1))
        self.assertEqual(model.patch_num_in, 12)
        self.assertEqual(model.patch_num_out, 12)
        self.assertEqual(
            model.time_branch.pattern_identifier.last_affinity_shape,
            (2, 12, 4),
        )
        self.assertEqual(model.time_branch.pattern_experts.last_topk_shape, (2, 12, 2))

        loss = output.square().mean() + model.auxiliary_loss()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_batch_size_one_and_multivariate_output_on_cpu(self):
        model = Model(make_config(c_out=7)).cpu().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 96, 7), None, None, None)
        self.assertEqual(tuple(output.shape), (1, 96, 7))
        self.assertEqual(output.device.type, 'cpu')

    def test_invalid_core_parameters_fail_fast(self):
        invalid_configs = [
            make_config(patch_len=97),
            make_config(tfps_t_top_k=5),
            make_config(d_model=15),
            make_config(tfps_t_num_experts=3),
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                Model(config)


if __name__ == '__main__':
    unittest.main()
