import csv
import time
import shutil
import threading
import subprocess
from pathlib import Path
from ultralytics import YOLO
import torch

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
TEST_RUN   = False         # True = prueba 5 épocas | False = entrenamiento real
BASE_DIR   = Path(__file__).parent
YAML_PATH  = str(BASE_DIR / "deeppcb.yaml")
PROJECT    = str(BASE_DIR / "runs" / "detect")

MODEL      = "yolo11s"
EPOCHS     = 5   if TEST_RUN else 100
BATCH      = 16
IMGSZ      = 800
WORKERS    = 4

# Nombre descriptivo con parámetros clave
_prefix    = "TEST" if TEST_RUN else "v2"
RUN_NAME   = f"{_prefix}_{MODEL}_ep{EPOCHS}_b{BATCH}_img{IMGSZ}"

# ─────────────────────────────────────────────
# MONITOREO GPU
# ─────────────────────────────────────────────
_stop_monitor = threading.Event()
_GPU_TEMP_LOG = str(BASE_DIR / "gpu_log_temp.csv")  # temporal hasta conocer run_dir real

def gpu_monitor(log_path: str):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gpu_util_%", "vram_used_mb",
                         "vram_total_mb", "gpu_temp_c", "power_w"])
        while not _stop_monitor.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,"
                    "temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"
                ], stderr=subprocess.DEVNULL).decode().strip()
                vals = [v.strip() for v in out.split(",")]
                writer.writerow([time.strftime("%H:%M:%S")] + vals)
                f.flush()
            except Exception:
                pass
            time.sleep(10)


def main():
    print(f"\n{'='*60}")
    print(f"  MODO   : {'PRUEBA' if TEST_RUN else 'ENTRENAMIENTO REAL'}")
    print(f"  Modelo : {MODEL}.pt")
    print(f"  Run    : {RUN_NAME}")
    print(f"  Épocas : {EPOCHS} | Batch: {BATCH} | Imgsz: {IMGSZ}")
    print(f"  YAML   : {YAML_PATH}")
    print(f"{'='*60}\n")

    # Monitoreo GPU — escribe en archivo temporal
    monitor_thread = threading.Thread(
        target=gpu_monitor, args=(_GPU_TEMP_LOG,), daemon=True
    )
    monitor_thread.start()
    print(f"Monitoreo GPU iniciado → {_GPU_TEMP_LOG}")

    t_start = time.time()
    print(f"Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Entrenamiento ──────────────────────────────────────────
    model = YOLO(f"{MODEL}.pt")
    results = model.train(
        data      = YAML_PATH,
        epochs    = EPOCHS,
        batch     = BATCH,
        imgsz     = IMGSZ,
        device    = 0,
        half      = True,
        workers   = WORKERS,
        project   = PROJECT,
        name      = RUN_NAME,
        exist_ok  = False,
        save      = True,
        plots     = True,
        val       = True,
        hsv_h     = 0.0,
        hsv_s     = 0.0,
        hsv_v     = 0.4,
        bgr       = 0.0,
    )

    # Directorio real usado por Ultralytics
    run_dir = Path(results.save_dir)

    # Timestamp fin
    t_end = time.time()
    total_time = t_end - t_start
    print(f"\nFin: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tiempo total: {total_time/60:.1f} min ({total_time:.0f} s)")

    # Detener monitoreo y mover gpu_log al directorio real del run
    _stop_monitor.set()
    monitor_thread.join(timeout=15)
    gpu_log_final = run_dir / "gpu_log.csv"
    shutil.move(_GPU_TEMP_LOG, str(gpu_log_final))
    print(f"gpu_log.csv movido a: {gpu_log_final}")

    # VRAM máxima
    vram_max_mb = torch.cuda.max_memory_allocated(0) / 1024**2
    print(f"VRAM máxima usada: {vram_max_mb:.0f} MB")

    # ── Evaluación final sobre TEST set ───────────────────────
    print("\n" + "="*60)
    print("  EVALUACIÓN FINAL sobre conjunto TEST")
    print("="*60)

    best_weights = run_dir / "weights" / "best.pt"
    model_eval = YOLO(str(best_weights))

    test_metrics = model_eval.val(
        data     = YAML_PATH,
        split    = "test",
        imgsz    = IMGSZ,
        device   = 0,
        half     = True,
        plots    = True,
        save_json= True,
        project  = str(run_dir),       # guarda dentro del run
        name     = "evaluacion_test",  # subcarpeta dentro del run
    )

    # ── Métricas globales ──────────────────────────────────────
    map50     = test_metrics.box.map50
    map5095   = test_metrics.box.map
    prec      = test_metrics.box.mp
    rec       = test_metrics.box.mr
    f1_global = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n{'─'*40}")
    print(f"  mAP50    : {map50:.4f}")
    print(f"  mAP50-95 : {map5095:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 global: {f1_global:.4f}")
    print(f"{'─'*40}")

    # ── Métricas por clase → CSV ───────────────────────────────
    class_names    = list(model_eval.names.values())
    ap50_per_class = test_metrics.box.ap50
    p_per_class    = test_metrics.box.p
    r_per_class    = test_metrics.box.r

    class_csv = run_dir / "metricas_por_clase.csv"
    with open(class_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clase", "AP50", "Precision", "Recall", "F1"])
        print(f"\n{'Clase':<12} {'AP50':>8} {'P':>8} {'R':>8} {'F1':>8}")
        print("─" * 48)
        for i, name in enumerate(class_names):
            p_i  = float(p_per_class[i])
            r_i  = float(r_per_class[i])
            ap_i = float(ap50_per_class[i])
            f1_i = 2 * p_i * r_i / (p_i + r_i) if (p_i + r_i) > 0 else 0.0
            writer.writerow([name, f"{ap_i:.4f}", f"{p_i:.4f}", f"{r_i:.4f}", f"{f1_i:.4f}"])
            print(f"{name:<12} {ap_i:>8.4f} {p_i:>8.4f} {r_i:>8.4f} {f1_i:>8.4f}")

    # ── Resumen computacional → CSV ────────────────────────────
    summary_csv = run_dir / "resumen_computacional.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metrica", "valor"])
        writer.writerow(["modelo", f"{MODEL}.pt"])
        writer.writerow(["epochs", EPOCHS])
        writer.writerow(["batch_size", BATCH])
        writer.writerow(["imgsz", IMGSZ])
        writer.writerow(["half_fp16", True])
        writer.writerow(["tiempo_total_min", f"{total_time/60:.2f}"])
        writer.writerow(["tiempo_total_seg", f"{total_time:.0f}"])
        writer.writerow(["vram_max_mb", f"{vram_max_mb:.0f}"])
        writer.writerow(["mAP50_test", f"{map50:.4f}"])
        writer.writerow(["mAP50_95_test", f"{map5095:.4f}"])
        writer.writerow(["precision_test", f"{prec:.4f}"])
        writer.writerow(["recall_test", f"{rec:.4f}"])
        writer.writerow(["f1_global_test", f"{f1_global:.4f}"])

    print(f"\nTodos los resultados guardados en: {run_dir}")
    print("  gpu_log.csv")
    print("  metricas_por_clase.csv")
    print("  resumen_computacional.csv")
    print("  evaluacion_test/  ← curvas y matrices del test set")
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
