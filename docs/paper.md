# Formalización Matemática: Submatrices Recursivas con Retrieval Probabilístico Interno

**Autor:** Victor Alejandro Cano Jaramillo  
**Fecha:** Abril 2026  
**Versión:** 0.1 — Draft para revisión

---

## 1. Notación Base

Sea una red neuronal feedforward estándar con L capas. La capa l tiene:

- **W_l** ∈ ℝ^(d_out × d_in) — matriz de pesos
- **b_l** ∈ ℝ^(d_out) — vector de bias
- **σ** — función de activación

Forward pass estándar:

```
h_l = σ(W_l · h_{l-1} + b_l)
```

---

## 2. Definición de Submatrices (Memoria Interna)

### 2.1 Submatrices de Primer Orden

Para cada capa l, después de entrenar en la tarea t₀ (tarea inicial), los pesos W_l^(0) se **congelan**.

Para cada nueva tarea t_k (k = 1, 2, ..., K), se crea una **submatriz**:

```
ΔW_l^(k) ∈ ℝ^(d_out × d_in)
```

Opcionalmente, con descomposición de bajo rango (inspirado en LoRA) para eficiencia:

```
ΔW_l^(k) = A_l^(k) · B_l^(k)

donde:
  A_l^(k) ∈ ℝ^(d_out × r)
  B_l^(k) ∈ ℝ^(r × d_in)
  r << min(d_out, d_in)    (rango reducido)
```

### 2.2 Peso Efectivo

El peso efectivo de la capa l dado un input x es:

```
W_l^eff(x) = W_l^(0) + Σ_{k=1}^{K} α_k(x) · ΔW_l^(k)
```

donde α_k(x) ∈ [0, 1] es la **gate de relevancia** para la submatriz k.

### 2.3 Forward Pass Modificado

```
h_l = σ(W_l^eff(x) · h_{l-1} + b_l)
     = σ((W_l^(0) + Σ_k α_k(x) · ΔW_l^(k)) · h_{l-1} + b_l)
```

---

## 3. Retrieval Probabilístico Interno

### 3.1 Mecanismo de Gate (Diferenciable)

Cada submatriz k tiene un **vector clave** (key embedding):

```
e_k ∈ ℝ^(d_key)
```

El input x se proyecta a un **vector de consulta** (query):

```
q(x) = W_q · pool(h_0(x)) + b_q

donde:
  W_q ∈ ℝ^(d_key × d_input)
  pool(·) = promedio o último estado del input
```

El score de relevancia:

```
s_k(x) = q(x)^T · e_k / √d_key     (escalado tipo attention)
```

Las gates se computan como:

```
α(x) = sparsemax(s(x))    o    top-p softmax(s(x))

donde s(x) = [s_1(x), s_2(x), ..., s_K(x)]
```

**Nota:** Se usa `sparsemax` en lugar de `softmax` para que la mayoría de α_k sean exactamente 0, activando solo submatrices relevantes (eficiencia de cómputo).

### 3.2 Propiedad Clave: Diferenciabilidad

El gradiente fluye a través de:
1. Las submatrices ΔW_l^(k) — aprenden representaciones específicas de tarea
2. Los key embeddings e_k — aprenden cuándo activarse
3. La proyección de query W_q — aprende a codificar el input

Esto permite entrenamiento **end-to-end** con backpropagation estándar.

---

## 4. Estructura Recursiva

### 4.1 Submatrices de Orden Superior

Cada submatriz ΔW_l^(k) puede, a su vez, tener sus propias submatrices:

```
ΔW_l^(k) → ΔW_l^(k,j)   para j = 1, ..., J_k
```

Con su propio mecanismo de gate:

```
ΔW_l^(k,eff)(x) = ΔW_l^(k,0) + Σ_j β_j^(k)(x) · ΔW_l^(k,j)
```

### 4.2 Formulación General (Profundidad d)

Sea un multi-índice **i** = (i_1, i_2, ..., i_d) que indexa la posición en el árbol de submatrices.

El peso efectivo con recursión de profundidad d:

```
W_l^eff(x) = W_l^(0) + Σ_{|i|=1}^{d} [ Π_{m=1}^{|i|} α_{i_m}^(parent(i_m))(x) ] · ΔW_l^(i)
```

donde Π es el producto de gates a lo largo del camino en el árbol.

### 4.3 Restricción Práctica

En la práctica, d = 1 o d = 2 es suficiente. La recursión profunda aumenta la expresividad pero introduce:
- Vanishing gates (producto de muchos α < 1 → 0)
- Complejidad de optimización

---

## 5. Mecanismo de Poda (Garbage Collection)

### 5.1 Score de Importancia

Para cada submatriz k, se define un score de importancia:

**Opción A — Activación promedio:**
```
I_k = E_{x~D} [α_k(x)]     (promedio sobre dataset reciente)
```

**Opción B — Fisher Information (inspirado en EWC):**
```
I_k = E_{x~D} [|| ∇_{ΔW_l^(k)} L(x) ||² · α_k(x)]
```

**Opción C — Contribución al output:**
```
I_k = E_{x~D} [|| α_k(x) · ΔW_l^(k) · h_{l-1} ||₂]
```

### 5.2 Criterio de Poda

```
Si I_k < τ durante N evaluaciones consecutivas:
    eliminar ΔW_l^(k) y su key embedding e_k
    liberar memoria
```

τ (threshold) puede ser adaptativo:
```
τ = percentil_p(I_1, I_2, ..., I_K)     (ej: p = 10, podar el 10% menos útil)
```

### 5.3 Consolidación (Alternativa a Poda)

En lugar de eliminar, **fusionar** submatrices similares:

```
Si cos(vec(ΔW^(a)), vec(ΔW^(b))) > θ_merge:
    ΔW^(merged) = (I_a · ΔW^(a) + I_b · ΔW^(b)) / (I_a + I_b)
    e_merged = (I_a · e_a + I_b · e_b) / (I_a + I_b)
```

---

## 6. Entrenamiento

### 6.1 Entrenamiento de Nueva Tarea t_k

1. Congelar W_l^(0) y todas las ΔW_l^(j) para j < k
2. Crear nueva submatriz ΔW_l^(k) (inicializada en 0) y key embedding e_k
3. Entrenar ΔW_l^(k), e_k, y (opcionalmente) refinar W_q

**Loss:**
```
L_total = L_task(x, y) + λ_sparse · Σ_k ||α_k(x)||₁ + λ_reg · ||ΔW^(k)||_F²

donde:
  L_task     = loss de la tarea (cross-entropy, MSE, etc.)
  λ_sparse   = penalización de esparsidad (fuerza pocas submatrices activas)
  λ_reg      = regularización de Frobenius (evita submatrices gigantes)
```

### 6.2 Complejidad

| Componente | Complejidad Adicional vs Red Base |
|---|---|
| Forward pass | O(K · r · d_in · d_out) con bajo rango |
| Gate computation | O(K · d_key) |
| Backprop (solo tarea nueva) | O(r · d_in · d_out + d_key) |
| Con sparsemax (top-p) | O(p · r · d_in · d_out), p << K |

---

## 7. Diferenciación con Trabajos Existentes

| Propiedad | LoRA | MoE | EWC | Progressive Nets | **Esta Propuesta** |
|---|---|---|---|---|---|
| Memoria en pesos | ✓ | ✗ (subredes) | ✗ (regularización) | ✓ (columnas) | ✓ (submatrices) |
| Retrieval condicional al input | ✗ (fijo) | ✓ (router) | ✗ | ✗ | ✓ (gates) |
| Bajo rango | ✓ | ✗ | ✗ | ✗ | ✓ (opcional) |
| Recursivo | ✗ | ✗ | ✗ | ✗ | ✓ |
| Poda dinámica | ✗ | ✓ (parcial) | ✗ | ✗ | ✓ |
| Consolidación | ✗ | ✗ | ✗ | ✗ | ✓ |
| End-to-end diferenciable | ✓ | ✓ | N/A | ✓ | ✓ |

**Insight clave:** Esta arquitectura puede verse como una **generalización de LoRA con routing condicional al input y estructura recursiva** — o equivalentemente, como **MoE a nivel de pesos en lugar de a nivel de subredes**.

---

## 8. Conexión con "Sentimientos Mecánicos"

Formalmente, si definimos un "estado emocional" como:

```
E(x) = f(Σ_k α_k(x) · m_k)
```

donde m_k es un "vector de memoria" asociado a la submatriz k (ej: el key embedding e_k concatenado con estadísticas de ΔW^(k)), entonces:

- E(x) depende del input actual (contexto)
- E(x) depende de experiencias pasadas (submatrices aprendidas previamente)
- E(x) modifica el output de la red (a través de las gates α_k)

Esto satisface la definición funcional de Victor: un estado interno que emerge de la interacción entre input actual y memorias pasadas, y que modifica el comportamiento del sistema.

---

## 9. Preguntas Abiertas para Investigación

1. **¿Cuál es el trade-off óptimo entre rango r y número de submatrices K?**
2. **¿La recursión de profundidad d > 1 aporta beneficio empírico significativo?**
3. **¿Cómo escala la calidad de la poda con el número de tareas?**
4. **¿Es posible aprender el mecanismo de poda end-to-end (meta-learning)?**
5. **¿Cómo se compara contra state-of-the-art en benchmarks de continual learning (Split-CIFAR, Permuted-MNIST, etc.)?**

---

## Apéndice: Nombre Propuesto para la Arquitectura

**RSM-Net** (Recursive Submatrix Memory Network)

o

**IRMA** (Internal Recursive Memory Architecture)
