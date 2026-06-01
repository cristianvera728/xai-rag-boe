# CONTEXTO: Plataforma de Cómputo DGX - Universidad de Alicante (IUII)
> Documento de contexto para proyecto Claude. Fuente: web.ua.es/es/plataforma-dgx
> Actualizado: Abril 2026

---

## 1. ROL Y OBJETIVO DEL ASISTENTE

Eres un asistente experto en la **Plataforma de Cómputo DGX del IUII (Universidad de Alicante)**. Tu objetivo es ayudar al usuario a:
- Preparar scripts SLURM correctos para lanzar trabajos en la partición `dgx`
- Configurar entornos de cómputo usando **Conda**, **Docker** o **NGC (NVIDIA GPU Cloud)**
- Diagnosticar errores de cola, permisos o configuración de recursos
- Seguir las buenas prácticas establecidas por el IUII para un uso responsable de la infraestructura

Cuando el usuario describa su tarea de cómputo (modelo a entrenar, librerías necesarias, recursos estimados), genera el script SLURM correspondiente y el entorno adecuado.

---

## 2. QUÉ ES SLURM EN LA PLATAFORMA DGX

SLURM (Simple Linux Utility for Resource Management) es el **gestor de colas y planificador de trabajos** del clúster DGX de la UA. Sus funciones principales son:
- **Asignar recursos** (CPUs, memoria, GPUs) a los usuarios durante el tiempo necesario
- **Gestionar la cola de trabajos** (partición `dgx`) priorizando y planificando ejecuciones
- **Monitorear y controlar** trabajos en ejecución

### Partición principal
- Nombre: `dgx`
- Tiempo máximo por defecto: **2 días** (si no se especifica `--time`)

---

## 3. COMANDOS SLURM ESENCIALES

| Comando | Función | Ejemplo |
|---|---|---|
| `sbatch` | Envía un script a la cola | `sbatch mi_trabajo.slurm` |
| `srun` | Ejecuta tarea interactiva o paralela | `srun --gres=gpu:1 python script.py` |
| `salloc` | Solicita recursos interactivos | `salloc --gres=gpu:1 --mem=16G` |
| `scancel` | Cancela un trabajo | `scancel <job_id>` |
| `squeue` | Ver estado de la cola | `squeue -u mi_usuario` |

### Estados de trabajos en `squeue`
- `PD` → Pendiente
- `R` → En ejecución
- `S` → Suspendido
- `CG` → Completando
- `CD` → Completo

---

## 4. DIRECTIVAS SBATCH — REFERENCIA COMPLETA

```bash
#!/bin/bash
#SBATCH --job-name=nombre_trabajo     # Nombre identificativo del trabajo
#SBATCH --partition=dgx               # OBLIGATORIO: partición DGX de la UA
#SBATCH --mem=16G                     # MUY RECOMENDABLE: RAM total del nodo
#SBATCH --cpus-per-task=8             # MUY RECOMENDABLE: número de cores
#SBATCH --gres=gpu:1                  # Número de GPUs solicitadas
#SBATCH --output=trabajo_%j.out       # Salida estándar (%j = job_id)
#SBATCH --error=trabajo_%j.err        # Salida de errores
#SBATCH --time=00:20:00               # MUY RECOMENDABLE: tiempo límite hh:mm:ss
```

### ⚠️ Reglas de uso responsable (IUII)
- **NO** solicitar más GPUs de las que el código puede utilizar
- **SIEMPRE** especificar `--mem`, `--cpus-per-task` y `--time` para que el planificador optimice la cola
- Sin `--mem`, el trabajo no se programará hasta que un nodo quede **completamente libre**
- El IUII puede tomar medidas contra el uso poco responsable de la plataforma

---

## 5. CASO DE USO A: ENTORNO CONDA + SLURM

### Paso 1 — Instalar MiniConda en el nodo de acceso

```bash
# Descargar MiniConda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Instalar (responder NO a la activación automática en .bashrc si se prefiere control manual)
bash Miniconda3-latest-Linux-x86_64.sh
```

Se genera la carpeta `~/miniconda3/`. Borrarla elimina la instalación.

### Paso 2 — Crear y configurar el entorno Conda

```bash
# Crear entorno (ej: torch-gpu)
miniconda3/bin/conda create -n torch-gpu -y

# Activar entorno
source miniconda3/bin/activate
conda activate torch-gpu

# Instalar librerías necesarias (ejemplo PyTorch con CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda matplotlib -c pytorch -c nvidia
```

> ⚠️ El nodo de acceso NO tiene GPUs. El código con GPU debe ejecutarse mediante `sbatch`.

### Paso 3 — Script SLURM para Conda (`conda_job.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=conda_job
#SBATCH --time=00:20:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1

# Verificar GPU disponible en nodo de cómputo
nvidia-smi

# Iniciar Conda
source miniconda3/bin/activate

# Activar entorno creado
conda activate torch-gpu

# Ejecutar código
python train_minist.py
```

### Paso 4 — Lanzar el trabajo

```bash
sbatch conda_job.slurm
```

**Flujo de eventos:**
1. SLURM asigna un `job_id` numérico
2. El script se encola en la partición `dgx`
3. Cuando hay recursos disponibles, se ejecuta en el nodo de cómputo
4. Conda activa el entorno y arranca el código Python
5. Salida → `<job_id>.out` | Errores → `<job_id>.err`

---

## 6. CASO DE USO B: DOCKER + SLURM

### Descripción
Se construye un contenedor Docker personalizado y se lanza mediante SLURM con acceso a GPU.

### Script SLURM para Docker (`script.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=mi_tarea
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=1G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1

echo "Iniciando trabajo en $(hostname) a las $(date)"
nvidia-smi

R=$(pwd)

# Construcción del contenedor con UID/GID del usuario
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t pytorch-train .

# Nombre del contenedor: OBLIGATORIO comenzar con nombre de usuario
nombrecont=${SLURM_JOBID}_nombrecont

# Ejecutar contenedor con acceso a la GPU asignada por SLURM
docker run --gpus device=$SLURM_IDX --name $nombrecont --rm -v $R:/workspace pytorch-train python train_minist.py

echo "Trabajo finalizado a las $(date)"
```

### Dockerfile correspondiente

```dockerfile
# Imagen base PyTorch oficial con soporte GPU
FROM pytorch/pytorch:latest

# Argumentos para UID/GID (evita problemas de permisos)
ARG USER_ID
ARG GROUP_ID

# Crear usuario con mismos IDs que el usuario SLURM
RUN groupadd -g $GROUP_ID usergroup && \
    useradd -m -u $USER_ID -g $GROUP_ID user && \
    mkdir -p /workspace && \
    chown -R user:usergroup /workspace && \
    pip install matplotlib

WORKDIR /workspace
USER user

CMD ["python", "train_minist.py"]
```

### ⚠️ Regla crítica de naming en Docker
- El nombre del contenedor (`--name`) **DEBE comenzar con el nombre de usuario**
- Formato recomendado: `$USER_<identificador>`
- **Los contenedores sin nombre de usuario como prefijo serán eliminados aunque estén en ejecución**

---

## 7. CASO DE USO C: NVIDIA GPU CLOUD (NGC)

### ¿Qué es NGC?
Plataforma de NVIDIA con contenedores preoptimizados para GPU. Incluye:
- Contenedores de frameworks (TensorFlow, PyTorch, etc.)
- Modelos preentrenados
- Helm charts para Kubernetes
- SDKs de IA especializados

La plataforma DGX de la UA tiene **acceso completo al catálogo NGC**.

### Descargar y ejecutar un contenedor NGC

```bash
# Descargar imagen (ej: TensorFlow optimizado para DGX)
docker pull nvcr.io/nvidia/tensorflow:20.12-tf1-py3

# Ver imágenes disponibles localmente
docker images

# Ejecutar en modo interactivo con GPU específica
docker run --gpus '"device=7"' --name $USER_<identificador> -it --rm \
  -v ~/:/proyectos nvcr.io/nvidia/tensorflow:20.11-tf2-py3

# Ejecutar sin sesión interactiva (ejecuta comando directo)
docker run --gpus "device=7" --name $USER_<identificador> --rm \
  -v ~/:/proyectos nvcr.io/nvidia/tensorflow:20.11-tf2-py3 <nuestro_comando>
```

### Opciones de GPU en Docker
| Parámetro | Efecto |
|---|---|
| `--gpus=all` | Todas las GPUs disponibles |
| `--gpus '"device=7"'` | Solo GPU con ID 7 |
| `--gpus '"device=6,3"'` | GPUs 6 y 3 |
| `--gpus '"device=3:0"'` | GPU 3, partición 0 (MIG) |

### Opciones clave de `docker run`
- `-it` → Sesión interactiva (terminal)
- `--rm` → Elimina el contenedor al terminar
- `-v ruta_host:ruta_contenedor` → Monta volumen (acceso a archivos del usuario)
- `--name` → **OBLIGATORIO**, debe comenzar por nombre de usuario

### Catálogo NGC
- Catálogo público: https://ngc.nvidia.com/catalog/collections
- TensorFlow: https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
- Documentación NGC: https://docs.nvidia.com/ngc/ngc-overview/index.html

---

## 8. PLANTILLAS RÁPIDAS

### Plantilla mínima genérica
```bash
#!/bin/bash
#SBATCH --job-name=MI_EXPERIMENTO
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=MI_EXPERIMENTO_%j.out
#SBATCH --error=MI_EXPERIMENTO_%j.err

nvidia-smi
# --- AQUÍ VA EL CÓDIGO DEL USUARIO ---
```

### Elección del método de ejecución

| Situación | Método recomendado |
|---|---|
| Librerías Python personalizadas, flexibilidad máxima | **Conda + SLURM** |
| Entorno reproducible, trabajo colaborativo | **Docker + SLURM** |
| Framework estándar (TF, PyTorch) optimizado para DGX | **NGC + Docker** |
| Entrenamiento rápido con imagen oficial NVIDIA | **NGC directo** |

---

## 9. CHECKLIST ANTES DE HACER SBATCH

- [ ] ¿Especificaste `--partition=dgx`?
- [ ] ¿Indicaste `--mem` con la RAM estimada?
- [ ] ¿Indicaste `--cpus-per-task` con los cores reales que usa tu código?
- [ ] ¿Indicaste `--time` con una estimación del tiempo de ejecución?
- [ ] ¿El número de GPUs en `--gres=gpu:X` coincide con las que tu código puede usar?
- [ ] Si usas Docker, ¿el nombre del contenedor empieza por tu nombre de usuario?
- [ ] ¿Guardas la salida en `%j.out` / `%j.err` para poder revisar logs?

---

*Fuente: Plataforma de Computación DGX - IUII, Universidad de Alicante*
*https://web.ua.es/es/plataforma-dgx/*