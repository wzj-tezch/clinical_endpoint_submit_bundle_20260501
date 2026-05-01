# 临床可读汇报 — endpoint 二分类

对照提示：若以此前 **`outputs_deep_full_gpu_sprint3`** 中用「≤3年时间窗二元代理标签」训练的树/MLP 为基线，本轮已将监督标签统一为 **`endpoint` 本体二分类**（仍保留 Cox 与时间依赖 AUC 作为辅助解读）。在同一外层划分与可比 seed 设置下，应重点对比 **`metrics_endpoint_binary_test.csv`**（AUC/BalancedAcc/F1 @0.5、Brier/ECE）与 **`clinical_thresholds_test_performance.csv`**。

## 阈值推荐（训练折 OOF 上定义；外测复述）
- 主方案：`max_balanced_accuracy`。
- 备选：`max_youden_j`、`conservative_high_specificity`、`aggressive_high_sensitivity`。

### 训练侧 OOF 上各规则对应的阈值与指标摘要
```
 threshold  balanced_accuracy       f1       mcc  sensitivity  specificity      ppv      npv   lr_pos   lr_neg  youden_j                          rule                     rule_name
  0.147068           0.543556 0.150470  0.045764     0.558140     0.528972 0.086957 0.937086 1.184939 0.835319  0.087111         max_balanced_accuracy         max_balanced_accuracy
  0.147068           0.543556 0.150470  0.045764     0.558140     0.528972 0.086957 0.937086 1.184939 0.835319  0.087111                  max_youden_j                  max_youden_j
  0.028840           0.499609 0.135524 -0.000487     0.767442     0.231776 0.074324 0.925373 0.998981 1.003376 -0.000782 conservative_high_specificity conservative_high_specificity
  0.147068           0.543556 0.150470  0.045764     0.558140     0.528972 0.086957 0.937086 1.184939 0.835319  0.087111   aggressive_high_sensitivity   aggressive_high_sensitivity
```

### 固定外测（复述各阈值对应的 Sens/Spec/PPV/NPV/LR+/LR-/F1/MCC）
```
{thr_tab}
```

## 概率层面的 endpoint 判别（阈值=0.5 及校准）
```
                   model      auc  balanced_accuracy_050   f1_050  mcc_050    brier      ece  calibration_slope_logit  calibration_intercept_logit
    model4_tree_endpoint 0.672285               0.533333 0.125000 0.248607 0.062894 0.046966                 0.441758                    -0.663612
     model5_mlp_endpoint 0.578652               0.576779 0.192308 0.104440 0.144998 0.205115                 0.126075                    -2.273181
model6_ensemble_endpoint 0.578652               0.576779 0.192308 0.104440 0.144998 0.205115                 0.126075                    -2.273181
```

## 图示
- `figures/calibration_curve_endpoint_ensemble.png`
- `figures/decision_curve_endpoint_ensemble.png`
（DCA 与同图所用概率为固定外测上的模型融合概率。）
