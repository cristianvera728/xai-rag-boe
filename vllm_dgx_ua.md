# vLLM en la plataforma DGX del IUII (Universidad de Alicante)

> Documento técnico de viabilidad y guía de despliegue.  
> Basado en la documentación oficial de la plataforma DGX-UA y el patrón de túnel SSH documentado en `llm_api_dgx_via_tunel.md`.

---

## ¿Es viable ejecutar vLLM aquí?

**Sí, es técnicamente viable.** vLLM puede desplegarse como API REST temporal en la plataforma DGX de la UA combinando un trabajo SLURM con un túnel SSH inverso. Las sesiones están limitadas a un máximo de 48 horas por trabajo.

---

## Arquitectura general

El acceso a la API sigue un patrón de doble túnel SSH, necesario porque los nodos de cómputo no son accesibles directamente desde el exterior:

```
Tu máquina          Login node (dgx.ua.es)       Nodo de cómputo (SLURM)
─────────────       ──────────────────────       ──────────────────────────
localhost:8000 ←──── puerto 18765 ←──────────────── vLLM server :8000
               ssh -L            ssh -R (abierto por el job al arrancar)
```

**Flujo completo:**

1. Lanzas el job con `sbatch vllm.slurm`
2. SLURM asigna un nodo de cómputo con GPU
3. El job arranca vLLM y abre un túnel inverso hacia el login node en el puerto 18765
4. Desde tu máquina abres el túnel forward: `ssh -L 8000:localhost:18765 dgx.ua.es`
5. Accedes a la API en `http://localhost:8000`

> **Prerequisito imprescindible:** SSH key configurada en `~/.ssh/` del NFS home del DGX. Sin ella el túnel inverso no puede abrirse desde el nodo de cómputo.

---

## Métodos de instalación

Tienes dos caminos compatibles con la plataforma DGX-UA:

### Opción A — Conda + pip (más simple)

El entorno Conda vive en `~/miniconda3/`, montado por NFS, por lo que está disponible en cualquier nodo de cómputo sin necesidad de copiar nada.

```bash
# En el nodo de login
source ~/miniconda3/bin/activate
conda create -n vllm-env python=3.11 -y
conda activate vllm-env
pip install vllm
```

### Opción B — Contenedor NGC (más optimizado)

NVIDIA publica contenedores oficiales de vLLM optimizados para DGX. La plataforma UA tiene acceso completo al catálogo NGC.

```bash
docker pull nvcr.io/nvidia/vllm:latest
```

> ⚠️ **Regla crítica de naming en Docker:** el nombre del contenedor **debe empezar por tu nombre de usuario**. Los contenedores sin este prefijo pueden ser eliminados por el IUII aunque estén en ejecución. Formato obligatorio: `$USER_vllm` o similar.

---

## Scripts SLURM

### Método A — Conda

```bash
#!/bin/bash
#SBATCH --job-name=vllm_api
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                  # Ajustar según el modelo (ver tabla más abajo)
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00            # Máximo permitido: 2 días
#SBATCH --output=vllm_%j.out
#SBATCH --error=vllm_%j.err

echo "Job iniciado en $(hostname) a las $(date)"
nvidia-smi

# Activar entorno
source ~/miniconda3/bin/activate
conda activate vllm-env

# Abrir túnel inverso hacia el login node en segundo plano
# (requiere SSH key configurada en ~/.ssh/)
LOGIN_NODE="dgx.ua.es"
REMOTE_PORT=18765
ssh -fNR ${REMOTE_PORT}:localhost:8000 ${LOGIN_NODE}

echo "Túnel inverso abierto en ${LOGIN_NODE}:${REMOTE_PORT}"
echo "Conectar con: ssh -L 8000:localhost:${REMOTE_PORT} ${LOGIN_NODE}"

# Lanzar servidor vLLM (bloqueante hasta que SLURM cancele el job)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

### Método B — NGC (Docker)

```bash
#!/bin/bash
#SBATCH --job-name=vllm_api
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=vllm_%j.out
#SBATCH --error=vllm_%j.err

echo "Job iniciado en $(hostname) a las $(date)"
nvidia-smi

# Nombre del contenedor: OBLIGATORIO empezar por nombre de usuario
CONT_NAME="${USER}_vllm_${SLURM_JOBID}"
LOGIN_NODE="dgx.ua.es"
REMOTE_PORT=18765

# Abrir túnel inverso
ssh -fNR ${REMOTE_PORT}:localhost:8000 ${LOGIN_NODE}
echo "Túnel inverso abierto. Conectar con: ssh -L 8000:localhost:${REMOTE_PORT} ${LOGIN_NODE}"

# Ejecutar vLLM via NGC con acceso a la GPU asignada por SLURM
docker run \
    --gpus device=${SLURM_IDX} \
    --name ${CONT_NAME} \
    --rm \
    -v ~/:/root \
    -p 8000:8000 \
    nvcr.io/nvidia/vllm:latest \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1
```

---

## Cómo conectarse desde tu máquina

Una vez el job esté en estado `R` (Running), ejecuta en tu terminal local:

```bash
ssh -L 8000:localhost:18765 tu_usuario@dgx.ua.es
```

A partir de ese momento la API está disponible en `http://localhost:8000`. Puedes verificarla con:

```bash
curl http://localhost:8000/v1/models
```

Y llamarla como cualquier API compatible con OpenAI:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-local"   # vLLM acepta cualquier string si no hay auth configurada
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hola, ¿funcionas?"}]
)
print(response.choices[0].message.content)
```

---

## Recursos necesarios según tamaño de modelo

| Modelo | VRAM mínima | `--mem` recomendado | GPUs |
|--------|------------|---------------------|------|
| 7B (Llama 3.1 8B, Mistral 7B) | ~16 GB | 32G | 1 |
| 13B | ~26 GB | 64G | 1 |
| 70B | ~140 GB | 128G | 2+ |

> Los modelos en formato GGUF cuantizado (4-bit) reducen la VRAM a la mitad aproximadamente, pero requieren backends adicionales.

---

## Almacenamiento del modelo

Los pesos del modelo deben descargarse antes o durante el job. El directorio home `~/` está montado por NFS y es accesible desde todos los nodos.

```bash
# En el nodo de login, antes de lanzar el job
pip install huggingface_hub
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ~/modelos/llama3-8b
```

Verifica tu cuota de disco disponible antes de descargar:

```bash
du -sh ~/
```

Los modelos de 7B-8B ocupan entre 14 y 16 GB en precisión BF16.

---

## Restricciones y riesgos a tener en cuenta

### ⏱ Límite de tiempo: 48 horas

La partición `dgx` permite un máximo de 2 días por trabajo. vLLM sólo puede funcionar como **API temporal**, no como servicio persistente. Si el job expira, la API deja de estar disponible y hay que relanzarlo.

### ⚡ GPU en estado idle — riesgo de política de uso

Este es el punto más delicado. El IUII puede tomar medidas contra el **uso irresponsable** de la plataforma. Un servidor vLLM mantiene la GPU asignada aunque no esté procesando peticiones. Para minimizar el riesgo:

- Limita `--time` al mínimo necesario para tu sesión de trabajo.
- No lances el job si no vas a utilizarlo activamente.
- Usa modelos que justifiquen el uso de GPU (no corras un modelo de 1B en una A100).
- Considera comunicárselo al IUII si planeas sesiones largas o recurrentes.

### 💾 Cuota de disco

Los modelos grandes ocupan decenas de GB en `~/`. La cuota del NFS puede ser un limitante. Elimina los pesos que ya no uses.

### 🔑 SSH key obligatoria

El túnel inverso desde el nodo de cómputo hacia el login node requiere que la SSH key esté en `~/.ssh/` del NFS. Sin ella el job arrancará vLLM pero no habrá forma de acceder a la API.

---

## Checklist antes de lanzar

- [ ] SSH key configurada en `~/.ssh/` del DGX
- [ ] Modelo descargado en `~/modelos/` (o se descargará durante el job)
- [ ] `--mem` ajustado al tamaño del modelo
- [ ] `--time` establecido (no dejar por defecto)
- [ ] Nombre del contenedor Docker empieza por tu usuario (si usas método B)
- [ ] Túnel forward abierto desde tu máquina tras confirmar que el job está en estado `R`

---

## Comparativa de métodos

| | Conda + pip | NGC container |
|---|---|---|
| Simplicidad | ✅ Alta | 🔶 Media |
| Rendimiento | 🔶 Estándar | ✅ Optimizado para DGX |
| Reproducibilidad | 🔶 Media | ✅ Alta |
| Naming rule Docker | No aplica | ⚠️ Obligatorio prefijo usuario |
| Tiempo de setup | ✅ Rápido (entorno ya existe) | 🔶 `docker pull` puede tardar |

---

*Fuente: documentación plataforma DGX-UA (web.ua.es/es/plataforma-dgx) + patrón llm_api_dgx_via_tunel.md*
