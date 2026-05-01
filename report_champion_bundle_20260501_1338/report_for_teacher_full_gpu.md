# 全量GPU深化实验终版报告

## 数据与协议
- 时间单位：天；endpoint作为事件标签（1=事件发生，0=未发生）。
- 评估时间窗：1y=365天，2y=730天，3y=1095天。
- 选入样本：266；总事件：45；3年事件：45。
- **树/MLP/融合**的监督标签为 `endpoint` 二分类（非 3 年窗代理）；内层 CV 优化目标：`0.4*AUC+0.3*BalancedAcc+0.3*F1`（概率阈值 0.5）。
- 评估口径：固定外层测试集 + 内层嵌套调参；同时报告原始 C-index、方向校正 C-index，以及 Cox 与时间依赖 AUC。
- OOF 上选定的融合阈值（Balanced Accuracy 主规则）：**0.258872**（详见 `clinical_thresholds_*.csv`）。
- 计算策略：requested=auto, selected=gpu, cpu_jobs=19。
- GPU状态：torch.cuda=True，xgboost可用=True。

- 树模型后端：xgboost（device=cuda）。
- MLP设备：cuda。

## Cox / 生存视角（固定外层测试）
```
               model  cindex_raw  cindex_adjusted  auc_365d  auc_730d  auc_1095d
     model1_base_cox    0.462963         0.537037  0.405844  0.405844   0.405844
     model2_geom_cox    0.555556         0.555556  0.409091  0.409091   0.409091
     model3_full_cox    0.592593         0.592593  0.524351  0.524351   0.524351
    model4_tree_auto    0.518519         0.518519  0.691558  0.691558   0.691558
     model5_mlp_auto    0.722222         0.722222  0.806818  0.806818   0.806818
model6_ensemble_auto    0.629630         0.629630  0.785714  0.785714   0.785714
```

## endpoint 二分类（固定外层测试 — 概率与阈值）
### 阈值（OOF→外测复述）与临床指标
```
 threshold  balanced_accuracy       f1      mcc  sensitivity  specificity      ppv      npv   lr_pos   lr_neg  youden_j           rule_name_train_oob  auc_probability
  0.258872           0.675325 0.411765 0.273573     0.636364     0.714286 0.304348 0.909091 2.227273 0.509091  0.350649         max_balanced_accuracy         0.785714
  0.258872           0.675325 0.411765 0.273573     0.636364     0.714286 0.304348 0.909091 2.227273 0.509091  0.350649                  max_youden_j         0.785714
  0.067562           0.676136 0.400000 0.263377     0.727273     0.625000 0.275862 0.921053 1.939394 0.436364  0.352273 conservative_high_specificity         0.785714
  0.130484           0.693994 0.421053 0.293019     0.727273     0.660714 0.296296 0.925000 2.143541 0.412776  0.387987   aggressive_high_sensitivity         0.785714
```

### 树 / MLP / 融合的概率型指标（含 Brier/ECE/logit校准）
```
                   model      auc  balanced_accuracy_050   f1_050  mcc_050    brier      ece  calibration_slope_logit  calibration_intercept_logit
    model4_tree_endpoint 0.691558               0.663961 0.470588 0.425423 0.119401 0.113435                 0.423730                    -0.685663
     model5_mlp_endpoint 0.806818               0.684253 0.424242 0.290682 0.172921 0.173543                 0.390884                    -1.201370
model6_ensemble_endpoint 0.785714               0.701299 0.480000 0.366825 0.131746 0.149403                 0.534991                    -0.827563
```

## 关键区间结果（bootstrap）
含 `*_auc_endpoint`、按 OOF 主阈值裁切的 `model6_ensemble_*_thr_oob`，以及 Cox/NRI。

```
                       metric       mean       ci2.5   ci97.5
     model1_base_cox_auc_1095   0.400899    0.216667 0.601628
      model1_base_cox_auc_365   0.400899    0.216667 0.601628
      model1_base_cox_auc_730   0.400899    0.216667 0.601628
 model1_base_cox_auc_endpoint   0.400899    0.216667 0.601628
         model1_base_cox_cadj   0.606271    0.504472 0.850179
         model1_base_cox_craw   0.452047    0.149821 0.733333
     model2_geom_cox_auc_1095   0.411725    0.230817 0.607229
      model2_geom_cox_auc_365   0.411725    0.230817 0.607229
      model2_geom_cox_auc_730   0.411725    0.230817 0.607229
 model2_geom_cox_auc_endpoint   0.411725    0.230817 0.607229
         model2_geom_cox_cadj   0.638356    0.507870 0.885794
         model2_geom_cox_craw   0.560063    0.221930 0.875000
       model2_geom_cox_idi_ep  -0.447009   -1.127384 0.020303
       model2_geom_cox_nri_ep  -0.249973   -0.850671 0.471264
     model3_full_cox_auc_1095   0.520079    0.336822 0.710604
      model3_full_cox_auc_365   0.520079    0.336822 0.710604
      model3_full_cox_auc_730   0.520079    0.336822 0.710604
 model3_full_cox_auc_endpoint   0.520079    0.336822 0.710604
         model3_full_cox_cadj   0.624415    0.500000 0.840014
         model3_full_cox_craw   0.586853    0.321393 0.840014
       model3_full_cox_idi_ep -86.265038 -284.511391 3.116642
       model3_full_cox_nri_ep   0.019440   -0.642677 0.714313
    model4_tree_auto_auc_1095   0.698845    0.485146 0.901009
     model4_tree_auto_auc_365   0.698845    0.485146 0.901009
     model4_tree_auto_auc_730   0.698845    0.485146 0.901009
model4_tree_auto_auc_endpoint   0.698845    0.485146 0.901009
        model4_tree_auto_cadj   0.595513    0.500000 0.795089
        model4_tree_auto_craw   0.519088    0.279818 0.777778
      model4_tree_auto_idi_ep   0.205138    0.024879 0.403538
      model4_tree_auto_nri_ep   0.000000    0.000000 0.000000
```

## 多seed复跑稳定性（固定外测）
含 endpoint AUC（`auc_endpoint`）与主阈值 Sens/Spec/BA/F1（`model6_*_thr_oob`）。

```
                                   metric     mean    ci2.5   ci97.5
                     model4_tree_auc_1095 0.693723 0.680925 0.705601
                 model4_tree_auc_endpoint 0.693723 0.680925 0.705601
                         model4_tree_cadj 0.567901 0.555556 0.590741
                         model4_tree_craw 0.567901 0.555556 0.590741
                      model5_mlp_auc_1095 0.794372 0.781737 0.803328
                  model5_mlp_auc_endpoint 0.794372 0.781737 0.803328
                          model5_mlp_cadj 0.716049 0.704630 0.722222
                          model5_mlp_craw 0.716049 0.704630 0.722222
                        model6_ba_thr_oob 0.678301 0.666843 0.692289
                      model6_ens_auc_1095 0.797078 0.794075 0.798701
                          model6_ens_cadj 0.635802 0.629630 0.647222
                          model6_ens_craw 0.635802 0.629630 0.647222
             model6_ensemble_auc_endpoint 0.797078 0.794075 0.798701
                        model6_f1_thr_oob 0.416422 0.400588 0.436213
                       model6_mcc_thr_oob 0.279739 0.257936 0.306783
                      model6_sens_thr_oob 0.636364 0.636364 0.636364
                      model6_spec_thr_oob 0.720238 0.697321 0.748214
threshold_used_from_main_oob_balanced_acc 0.258872 0.258872 0.258872
```

## 增益指标
- NRI/IDI（二元 **endpoint** 标签）已写入 `nri_idi_fixed_test.csv`。

## 解释性与定位
- seg14_amp: 118 次 / 200 次（59.00%）
- seg06_amp: 114 次 / 200 次（57.00%）
- seg16_amp: 93 次 / 200 次（46.50%）
- seg10_amp: 76 次 / 200 次（38.00%）
- seg11_amp: 69 次 / 200 次（34.50%）

## 图表与交付清单
- `figures/time_dependent_auc_curve.png`
- `figures/calibration_curve_endpoint_ensemble.png` / `figures/decision_curve_endpoint_ensemble.png`
- `figures/bullseye_standard_diff.png`
- `figures/bullseye_top5_stability.png`
- `metrics_endpoint_binary_test.csv` · `clinical_thresholds_test_performance.csv`
- `protocol_endpoint.json` · `report_clinical_endpoint.md`
- `seed_repeat_metrics.csv` / `bootstrap_metrics.csv` 等同 seed 可比文件。
