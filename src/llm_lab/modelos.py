"""
Catálogo de modelos de lenguaje locales y función de consulta común.

Todos los modelos se descargan de Hugging Face y se ejecutan en la GPU de Colab,
cuantizados a 4 bits (NF4). No se usa ningún servicio en línea: los datos no
salen de Colab. El token de Hugging Face (solo para descargar modelos con
licencia que aceptar) se lee de la variable de entorno HF_TOKEN o de los
secretos de Colab, nunca del código.

Solo hay un modelo cargado a la vez: liberar() vacía la GPU antes de cargar otro.
"""

import gc
import importlib.metadata as md
import os
import platform
import re
import time

# estado: "probado" = usado en los experimentos; el resto hay que comprobarlo antes de usarlo.
CATALOGO_MODELOS = {
    "qwen2.5-7b": {"hf": "Qwen/Qwen2.5-7B-Instruct", "gpu": "T4", "sistema": True, "plantilla": {},
                   "estado": "probado", "nota": "Modelo de la memoria (Experimentos 2 y 3)"},
    "qwen2.5-14b": {"hf": "Qwen/Qwen2.5-14B-Instruct", "gpu": "T4", "sistema": True, "plantilla": {},
                    "estado": "probado", "nota": "Misma generación, el doble de tamaño (Experimento 2)"},
    "qwen3.5-9b": {"hf": "Qwen/Qwen3.5-9B", "gpu": "T4", "sistema": True, "plantilla": {"enable_thinking": False},
                   "estado": "probado", "nota": "Generación más reciente; multimodal; razonamiento desactivado"},
    "qwen3.5-27b": {"hf": "Qwen/Qwen3.5-27B", "gpu": "L4", "sistema": True, "plantilla": {"enable_thinking": False},
                    "estado": "sin probar", "nota": "No cabe en la T4; multimodal; razonamiento desactivado"},
    "llama3.1-8b": {"hf": "meta-llama/Llama-3.1-8B-Instruct", "gpu": "T4", "sistema": True, "plantilla": {},
                    "estado": "sin probar", "nota": "Requiere aceptar la licencia y el token de Hugging Face"},
    "phi-4": {"hf": "microsoft/phi-4", "gpu": "T4", "sistema": True, "plantilla": {},
              "estado": "sin probar", "nota": "Orientado a razonamiento matemático"},
}


def token_hugging_face():
    """Token de Hugging Face desde HF_TOKEN o los secretos de Colab; None si no hay."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:  # fuera de Colab o sin el secreto
        return None


def entorno():
    """Versiones y GPU, para guardarlas en cada ejecución."""
    import torch
    datos = {"python": platform.python_version(),
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for paquete in ["numpy", "pandas", "torch", "transformers", "bitsandbytes", "accelerate"]:
        try:
            datos[paquete] = md.version(paquete)
        except md.PackageNotFoundError:
            datos[paquete] = None
    return datos


class ModeloLLM:
    """
    Un modelo del catálogo. Con cargar_pesos=False solo se carga el tokenizador
    (suficiente para montar prompts, contar tokens y comprobar huellas sin GPU).
    """

    def __init__(self, clave, cargar_pesos=True):
        from transformers import AutoTokenizer
        if clave not in CATALOGO_MODELOS:
            raise KeyError(f"Modelo desconocido: {clave}. Disponibles: {list(CATALOGO_MODELOS)}")
        self.clave, self.info = clave, CATALOGO_MODELOS[clave]
        self._token = token_hugging_face()
        self.tokenizador = AutoTokenizer.from_pretrained(self.info["hf"], token=self._token)
        self.modelo = None
        if cargar_pesos:
            self.cargar_pesos()

    def cargar_pesos(self):
        """Carga el modelo en la GPU, cuantizado a 4 bits NF4 con cómputo en float16."""
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        if not torch.cuda.is_available():
            print("Sin GPU: solo se ha cargado el tokenizador.")
            return
        configuracion = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16)
        self.modelo = AutoModelForCausalLM.from_pretrained(self.info["hf"], quantization_config=configuracion,
                                                           device_map="auto", token=self._token)
        print(f"{self.clave} cargado en {self.modelo.device}")

    def liberar(self):
        """Libera la GPU (antes de cargar otro modelo)."""
        import torch
        self.modelo = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def parametros(self):
        """Lo que identifica al modelo y la generación; se guarda en cada ejecución."""
        return {"modelo": self.clave, "modelo_hf": self.info["hf"], "cuantizacion": "4 bits NF4, cómputo float16",
                "do_sample": False, "plantilla": self.info["plantilla"]}

    def _adaptar(self, mensajes):
        # Modelos sin mensaje de sistema: el rol se antepone al mensaje de usuario
        if self.info["sistema"] or mensajes[0]["role"] != "system":
            return mensajes
        return [{"role": "user", "content": mensajes[0]["content"] + "\n\n" + mensajes[1]["content"]}] + mensajes[2:]

    def texto_formateado(self, mensajes):
        """Texto exacto que recibe el modelo (con su plantilla de chat)."""
        return self.tokenizador.apply_chat_template(self._adaptar(mensajes), add_generation_prompt=True, tokenize=False,
                                                    **self.info["plantilla"])

    def contar_tokens(self, texto):
        return len(self.tokenizador.encode(texto, add_special_tokens=False))

    def probabilidad_anomalia(self, tokens_generados, puntuaciones):
        """
        P("1") / (P("1") + P("0")) en el paso en que el modelo escribe el valor de "anomalous".
        None si no encuentra ese campo (respuesta con otro formato).
        """
        import torch
        tok = self.tokenizador
        for k, id_token in enumerate(tokens_generados.tolist()):
            if tok.decode([id_token]).strip() not in ("0", "1"):
                continue
            if not re.search(r'"anomalous"\s*:\s*$', tok.decode(tokens_generados[:k])):
                continue
            token = tok.convert_ids_to_tokens(id_token)
            id_uno = tok.convert_tokens_to_ids(token.replace("0", "1"))
            id_cero = tok.convert_tokens_to_ids(token.replace("1", "0"))
            probabilidades = torch.softmax(puntuaciones[k][0].float(), dim=-1)
            p_uno, p_cero = probabilidades[id_uno].item(), probabilidades[id_cero].item()
            return p_uno / (p_uno + p_cero)
        return None

    def consultar(self, mensajes, max_new_tokens):
        """
        Consulta con decodificación voraz (sin muestreo: la misma entrada da la misma salida).
        Devuelve la respuesta en bruto, el texto exacto que se tokenizó, tokens, tiempo y P(anomalía).
        `coincide_tokens` comprueba que ese texto produce exactamente los tokens que recibió el modelo.
        """
        import torch
        if self.modelo is None:
            raise RuntimeError("El modelo no está cargado (hace falta GPU)")
        tok, adaptados = self.tokenizador, self._adaptar(mensajes)
        entrada = tok.apply_chat_template(adaptados, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                          **self.info["plantilla"]).to(self.modelo.device)
        texto = self.texto_formateado(mensajes)
        n_entrada = entrada["input_ids"].shape[1]
        inicio = time.time()
        with torch.no_grad():
            salida = self.modelo.generate(**entrada, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=tok.eos_token_id, output_scores=True, return_dict_in_generate=True)
        generados = salida.sequences[0, n_entrada:]
        return {
            "respuesta_bruta": tok.decode(generados, skip_special_tokens=True),
            "texto_formateado": texto,
            "coincide_tokens": tok(texto, add_special_tokens=False)["input_ids"] == entrada["input_ids"][0].tolist(),
            "tokens_entrada": n_entrada,
            "tokens_salida": len(generados),
            "segundos": round(time.time() - inicio, 2),
            "p_interna": self.probabilidad_anomalia(generados, salida.scores),
        }
