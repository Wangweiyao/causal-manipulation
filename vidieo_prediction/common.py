# graph
nf = 64
edge_share = True
hard_gumble_softmax = True
edge_st_idx = 1
edge_pos_prior = 0.1

# embedding
z_what_dim = 8
z_where_scale_dim = 2  # sx sy
z_where_shift_dim = 2  # tx ty
z_pres_dim = 1
glimpse_size = 32
img_h = 64
img_w = img_h
img_encode_dim = 64
z_depth_dim = 1
bg_what_dim = 1
action_dim = 4

DEBUG = True

phase_obj_num_contrain = False
phase_rejection = True

z_where_bias_dim = 4
where_update_scale = .2
pres_logit_factor = 8.8

cfg = {
    'num_img_summary': 3,
    'num_cell_h': 4,
    'num_cell_w': 4,
    'phase_no_background': False,
    'static_background': True,
    'causal_factorization': True,
    'phase_eval': True,
    'phase_boundary_loss': False,
    'phase_generate': False,
    'phase_nll': False,
    'gen_disc_pres_probs': 0.1,
    'observe_frames': 2,
    'size_anc': 0.4,
    'var_s': 0.3,
    'ratio_anc': 1.0,
    'var_anc': 0.5,
    'color_num': 500,
    'explained_ratio_threshold': 0.1,
    'tau_imp': 0.25,
    'z_pres_anneal_end_value': 1e-3,
    'phase_do_remove_detach': False,
    'remove_detach_step': 30000,
    'max_num_obj': 4 # Remove this constrain in discovery.py if you have enough GPU memory.
}
