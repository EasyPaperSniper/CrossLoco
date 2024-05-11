from .assects_cfg import ALIENGOZ1_CFG, SMPLHUMAN_CFG, CHILD_CFG, ALIENGOARX5_CFG, GO2ARX5_CFG


SMPL_KEY_LINK_IDX = [0,
                     10, 11, 12,
                     22, 23, 24,
                     34, 35, 36,
                     52, 53, 55, 54, 56,
                     67, 66, 68,
                     75, 76,
                     83, 84]

GO2AR_KEY_LINK_IDX = [0,        # base
                      1, 2, 3, 4, 5, # FL_hip, FR_hip, Head, RL_hip, RR_hip
                      9, 10, 12, 13, # RL-thigh, FR-thigh, RL-thigh, RR-thigh
                      15, 16, 17, 18, # FL-calf, FR-calf, RL-calf, RR-calf
                      19, 21, 23, 25, 27, # link1, FL-foot, FR-foot, RL-foot, RR-foot 
                      28, 33, 34, 36 ] # link2, link3, link4, link6