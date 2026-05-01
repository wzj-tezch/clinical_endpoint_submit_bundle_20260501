Champion output copied by pick_report_champion.py
Source: E:\本基中间结果\outputs_clinical_endpoint_n320_bs400
Rule: max 0.4*AUC+0.3*balanced_accuracy_050+0.3*f1_050 on model6_ensemble_endpoint; tie-break mcc_050 then lower brier
See report_champion_selection.json in repo root for full ranking.

OOF Isotonic adjunct (same min-samples=320, bootstrap 400): outputs_clinical_endpoint_champion_n320_iso/
  Files: metrics_endpoint_isotonic_test.csv, clinical_thresholds_test_performance_isotonic.csv,
         figures/calibration_curve_endpoint_ensemble_isotonic.png, decision_curve_endpoint_ensemble_isotonic.png
  On this cohort isotonic lowers Brier/ECE vs raw fusion but discrimination can shift; cite as supplementary calibration pathway.
