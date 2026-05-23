import numpy as np
import pandas as pd
import re
from utils.logger import Logger
from utils.metrics import MetricsUtil

class ResultReporter:
    MODEL_ORDER = {
        "PLS": 1,
        "KRR": 2,
        "ElasticNet": 3,
        "RFRR": 4,
        "RSRidge": 5,
        "GlobalStack": 6,
        "MoE": 7
    }

    @staticmethod
    def print_shared_bands(shared_idx, wavelength_cols):
        shared_waves = wavelength_cols[shared_idx]
        shared_waves_float = pd.to_numeric(pd.Index(shared_waves), errors="coerce").values
        
        order_by_wave = np.argsort(shared_waves_float)
        Logger.log("[SharedBands] ===== Print in ascending order by wavelength =====")
        for k, oi in enumerate(order_by_wave, start=1):
            Logger.log(f"[SharedBands] #{k:02d} idx={shared_idx[oi]:4d} wave={shared_waves_float[oi]:.2f} nm (col='{shared_waves[oi]}')")

        return shared_waves, shared_waves_float

    @staticmethod
    def build_summary_from_fold_metrics(fold_metrics_df, rename_test_to_oof=True, model_filter=None):
        mean_metrics = fold_metrics_df.groupby(['Target', 'Model', 'Set'])[['R2', 'RMSE', 'RPD']].mean().reset_index()

        if model_filter is not None:
            mean_metrics = mean_metrics[mean_metrics["Model"] == model_filter].copy()

        pivot_df = mean_metrics.pivot(index=['Target', 'Model'], columns='Set', values=['R2', 'RMSE', 'RPD'])

        pivot_df.columns = [f"{col[1]} {col[0]}" for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        if rename_test_to_oof and "Test R2" in pivot_df.columns:
            pivot_df = pivot_df.rename(columns={
                "Test R2": "OOF R2",
                "Test RMSE": "OOF RMSE",
                "Test RPD": "OOF RPD"
            })

        test_prefix = "OOF" if rename_test_to_oof else "Test"
        cols_order = ['Target', 'Model', 'Train R2', 'Train RMSE', 'Train RPD',
                      f'{test_prefix} R2', f'{test_prefix} RMSE', f'{test_prefix} RPD']
        valid_cols = [c for c in cols_order if c in pivot_df.columns]
        df_summary = pivot_df[valid_cols]

        df_summary = df_summary.copy()
        df_summary['_sort_idx'] = df_summary['Model'].map(ResultReporter.MODEL_ORDER).fillna(99)
        df_summary = df_summary.sort_values(by=['Target', '_sort_idx']).drop(columns=['_sort_idx'])
        return df_summary

    @staticmethod
    def build_summary_from_oof_predictions(
        fold_metrics_df,
        target_metals,
        Y_true_matrix,
        oof_prediction_dicts,
        test_prefix="OOF",
        model_filter=None
    ):
        train_metrics = (
            fold_metrics_df[fold_metrics_df["Set"] == "Train"]
            .groupby(["Target", "Model"])[["R2", "RMSE", "RPD"]]
            .mean()
        )

        rows = []
        for model_name, pred_dict in oof_prediction_dicts.items():
            if model_filter is not None and model_name != model_filter:
                continue

            for j, metal in enumerate(target_metals):
                y_true = np.asarray(Y_true_matrix[:, j], dtype=float)
                y_pred = np.asarray(pred_dict[metal], dtype=float)
                valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
                if valid_mask.sum() != len(y_true):
                    raise ValueError(
                        f"OOF predictions for target='{metal}', model='{model_name}' "
                        "are incomplete or contain non-finite values."
                    )

                test_r2, test_rmse, test_rpd = MetricsUtil.compute_metrics(y_true, y_pred)
                row = {
                    "Target": metal,
                    "Model": model_name,
                    f"{test_prefix} R2": test_r2,
                    f"{test_prefix} RMSE": test_rmse,
                    f"{test_prefix} RPD": test_rpd
                }

                if (metal, model_name) in train_metrics.index:
                    tr = train_metrics.loc[(metal, model_name)]
                    row.update({
                        "Train R2": tr["R2"],
                        "Train RMSE": tr["RMSE"],
                        "Train RPD": tr["RPD"]
                    })

                rows.append(row)

        df_summary = pd.DataFrame(rows)
        if df_summary.empty:
            return df_summary

        cols_order = [
            "Target", "Model", "Train R2", "Train RMSE", "Train RPD",
            f"{test_prefix} R2", f"{test_prefix} RMSE", f"{test_prefix} RPD"
        ]
        valid_cols = [c for c in cols_order if c in df_summary.columns]
        df_summary = df_summary[valid_cols].copy()
        df_summary["_sort_idx"] = df_summary["Model"].map(ResultReporter.MODEL_ORDER).fillna(99)
        df_summary = df_summary.sort_values(by=["Target", "_sort_idx"]).drop(columns=["_sort_idx"])
        return df_summary

    @staticmethod
    def _sheet_name_for_target(target_name):
        match = re.search(r'([A-Z][a-z]?)', str(target_name))
        if match:
            return match.group(1)
        safe_name = re.sub(r'[\[\]\:\*\?\/\\]', '_', str(target_name)).strip()
        return safe_name[:31] or "Target"

    @staticmethod
    def export_oof_predictions_to_excel(output_path, sample_ids, outer_fold_ids, target_metals, Y_true_matrix, oof_pred_global_dict, oof_pred_moe_dict, fold_metrics_df):
        sample_ids = np.asarray(sample_ids)
        outer_fold_ids = np.asarray(outer_fold_ids)

        df_g = pd.DataFrame({"Sample": sample_ids, "OuterFold": outer_fold_ids.astype(int)})
        df_m = pd.DataFrame({"Sample": sample_ids, "OuterFold": outer_fold_ids.astype(int)})

        for j, metal in enumerate(target_metals):
            y_true = np.asarray(Y_true_matrix[:, j], dtype=float)
            pred_g = np.asarray(oof_pred_global_dict[metal], dtype=float)
            pred_m = np.asarray(oof_pred_moe_dict[metal], dtype=float)

            df_g[f"{metal}_true"] = y_true
            df_g[f"{metal}_pred"] = pred_g
            df_m[f"{metal}_true"] = y_true
            df_m[f"{metal}_pred"] = pred_m

        df_summary = ResultReporter.build_summary_from_oof_predictions(
            fold_metrics_df,
            target_metals,
            Y_true_matrix,
            {
                "GlobalStack": oof_pred_global_dict,
                "MoE": oof_pred_moe_dict
            },
            test_prefix="OOF",
            model_filter="MoE"
        )

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_g.to_excel(writer, sheet_name="GlobalStack_Predictions", index=False)
            df_m.to_excel(writer, sheet_name="MoE_Predictions", index=False)
            fold_metrics_df.to_excel(writer, sheet_name="Fold_Performance", index=False)
            df_summary.to_excel(writer, sheet_name="Global_Summary", index=False)

        Logger.log(f"[SAVE] OOF predictions and summaries saved to: {output_path}")

    @staticmethod
    def export_sweep_summary_to_excel(output_path, sweep_summary_df):
        df = sweep_summary_df.copy()
        if df.empty:
            raise ValueError("Sweep summary is empty, nothing to export.")

        df["_sort_idx"] = df["Model"].map(ResultReporter.MODEL_ORDER).fillna(99)
        sort_cols = ["Target", "SharedMaxFeatures", "_sort_idx"]
        sort_cols = [col for col in sort_cols if col in df.columns]
        df = df.sort_values(sort_cols)

        ordered_cols = [
            "SharedMaxFeatures",
            "Model",
            "Train R2",
            "Train RMSE",
            "Train RPD",
            "Test R2",
            "Test RMSE",
            "Test RPD"
        ]
        ordered_cols = [col for col in ordered_cols if col in df.columns]

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for target in df["Target"].drop_duplicates():
                target_df = df[df["Target"] == target].drop(columns=["_sort_idx"]).copy()
                target_df = target_df[ordered_cols]
                sheet_name = ResultReporter._sheet_name_for_target(target)
                target_df.to_excel(writer, sheet_name=sheet_name, index=False)

        Logger.log(f"[SAVE] Sweep summary saved to: {output_path}")
