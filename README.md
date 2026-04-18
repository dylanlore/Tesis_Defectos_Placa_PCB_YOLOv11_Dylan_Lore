# Tesis — Inspección Automática de Defectos en Placas PCB mediante YOLOv11

**Universidad Tecnológica de Panamá**
**Autor:** Dylan Loré
**Asesora:** Mgtr. Carmen Miranda Salazar

## Descripción

Entrenamiento y evaluación del modelo YOLOv11 para la detección automática de defectos en placas PCB (Printed Circuit Boards), utilizando el dataset DeepPCB y hardware de gama media (RTX 3050 4GB), con enfoque de eficiencia económica y computacional.

## Dataset

**DeepPCB** — 1,500 imágenes (640×640 px), 6 clases de defectos:

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | open | Circuito abierto |
| 1 | short | Cortocircuito |
| 2 | mousebite | Mordida de ratón |
| 3 | spur | Espolón |
| 4 | copper | Cobre espurio |
| 5 | pin-hole | Agujero de alfiler |

## Estructura del repositorio

Cada rama corresponde a una versión de entrenamiento con sus propios parámetros y resultados:

```
main                          → estructura base y documentación
v1-yolo11s-ep100-b8-img640   → versión 1: configuración base
v2-...                        → versión 2: parámetros modificados
```

## Hardware

- CPU: AMD Ryzen 5 6600H
- GPU: NVIDIA GeForce RTX 3050 (4 GB VRAM)
- RAM: 16 GB
- CUDA: 12.4

## Stack tecnológico

- Python 3.12
- PyTorch 2.6.0+cu124
- Ultralytics 8.4.21 (YOLOv11)
- OpenCV, NumPy, Matplotlib
