import pandas as pd
import os
import types
import numpy as np
from dataClasses import *

'''
# Ruta relativa al archivo dentro de la carpeta "data"
base_dir = os.path.dirname(os.path.abspath(__file__))  # Carpeta donde está `filtro.py`
ruta_archivo_excel = os.path.join(base_dir, "data", "filtered3.xlsx")

# Verificar si el archivo existe
if not os.path.exists(ruta_archivo_excel):
    raise FileNotFoundError(f"El archivo '{ruta_archivo_excel}' no se encuentra en la carpeta 'data'.")

# Leer el archivo Excel
df = pd.read_excel(ruta_archivo_excel)

# Validar que las columnas necesarias existan
columna_precio = "Price in (1M Tokens)"
columna_empresa = "Organization"
columna_modelo = "Model"

for columna in [columna_precio, columna_empresa, columna_modelo]:
    if columna not in df.columns:
        raise KeyError(f"La columna '{columna}' no existe en el archivo Excel.")

# Limpiar la columna `Price in (1M Tokens)` y convertir valores a numéricos
df[columna_precio] = pd.to_numeric(df[columna_precio], errors="coerce").fillna(0)

# Crear dinámicamente un método para cada modelo LLM
def metodo_llm_factory(precio, modelo):
    def metodo_llm(self):
        """
        Método dinámico que devuelve los tokens asociados al modelo.
        """
        return f"Modelo: {modelo}, Tokens asociados: {precio}"
    return metodo_llm

# Crear clases dinámicas con nombres reales de las empresas
clases_por_empresa = {
    "01AI.__name__": a_01AI,
    "AI21_Labs.__name__": AI21Labs,
    "Alibaba.__name__": Alibaba,
    "Anthropic.__name__": Anthropic,
    "Cohere.__name__": Cohere,
    "DeepSeek.__name__": DeepSeek,
    "DeepSeekAI.__name__": DeepSeekAI,
    "Google.__name__": Google,
    "Meta.__name__": Meta,
    "Mistral.__name__": Mistral,
    "NexusFlow.__name__": NexusFlow,
    "Nvidia.__name__": Nvidia,
    "OpenAI.__name__": OpenAI,
    "Princeton.__name__": Princeton,
    "Reka.__name__": RekaAI,
    "Zhipu.__name__": ZhipuAI,
}


for index, fila in df.iterrows():
    empresa = fila[columna_empresa]  # Nombre de la empresa
    modelo_llm = fila[columna_modelo]  # Nombre del modelo
    precio = fila[columna_precio]  # Tokens (precio) asociado

    # Verificar si la empresa ya tiene una clase generada
    if empresa not in clases_por_empresa:
        # Crear una nueva clase para la empresa
        class Empresa:
            def __init__(self, nombre):
                self.nombre = nombre

        Empresa.__name__ = empresa
        clases_por_empresa[empresa] = Empresa(empresa)

    # Crear el método dinámico para el modelo
    nombre_metodo = ''.join([palabra.capitalize() for palabra in modelo_llm.split('_')])
    metodo = metodo_llm_factory(precio, modelo_llm)
    setattr(clases_por_empresa[empresa], nombre_metodo, types.MethodType(metodo, clases_por_empresa[empresa]))

# Iterar sobre las clases dinámicas y mostrar los tokens asociados a cada modelo
for empresa, instancia in clases_por_empresa.items():
    print(f"\nEmpresa: {empresa}")
    for metodo in dir(instancia):
        if not metodo.startswith("__") and callable(getattr(instancia, metodo)):  # Ignorar métodos internos
            metodo_func = getattr(instancia, metodo)
            print(f"Resultado del método {metodo}: {metodo_func()}")

'''

if __name__ == "__main__":
    
    input_string = "Este es un ejemplo de cadena de texto."

    openAI=OpenAI()
    print("OpenAI")
    print(f"El numero de tokens es: {openAI.ChatGPT4oLatest(input_string)}")
    print(f"El numero de tokens es: {openAI.GPT4omini20240718(input_string)}")
    print(f"El numero de tokens es: {openAI.o1_preview(input_string)}")

    google=Google()
    print("Google")
    print(f"El numero de tokens es: {google.Gemini_1_5_Pro_002(input_string)}")
    print(f"El numero de tokens es: {google.Gemini_1_5_Flash_Exp_0827(input_string)}")
    print(f"El numero de tokens es: {google.Gemma_2_27b_it(input_string)}")


    princeton=Princeton()
    print("Princeton")
    print(f"El numero de tokens es: {princeton.Gemma_2_9b_it_SimPO(input_string)}")

    nvidia=Nvidia()
    deepSeek=DeepSeek()
    deepSeekAI=DeepSeekAI()
    anthropic=Anthropic()
    cohere=Cohere()
    zhipu=zhipu_AI()
    meta=Meta()
    b_01AI=a_01AI()
    aI21Labs=AI21Labs()




    alibaba=Alibaba()
    print("Alibaba")
    print(f"El numero de tokens es: {alibaba.Qwen_Max_0919(input_string)}")
    print(f"El numero de tokens es: {alibaba.Qwen2_5_72b_Instruct(input_string)}")
    print(f"El numero de tokens es: {alibaba.Qwen_Plus_0828(input_string)}")