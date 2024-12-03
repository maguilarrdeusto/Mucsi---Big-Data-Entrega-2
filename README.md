## Descripción del problema
Para el desarrollo de este caso, se planteó un caso de negocios que busca integrar una
solución para problemas de post-venta y/o atención al cliente, centrada en la autogestión
de problemáticas para agilizar los tiempos de gestión y respuesta.

De esta manera, se plantea la necesidad de encontrar una solución que facilite la
orientación de la información mediante la agrupación de esta en un formato que facilite su
consulta y rápida respuesta. Para este caso se planteó el uso de una base de
conocimientos ficticia en formato texto, la que se lleva hacía una respuesta generativa que
oriente al cliente antes de poder levantar su requerimiento.

La vectorización de esta información se fundamenta en la utilización de embeddings,
recurso que permite transformar textos en representaciones numéricas de alta
dimensionalidad, las cuales capturan el significado semántico de los datos más allá de las
simples coincidencias de palabras clave. Este enfoque habilita la búsqueda semántica y la
comparación de similitud entre consultas y documentos de manera eficiente, incluso en
conjuntos de datos heterogéneos o con terminología especializada.

Para esto, se debe comprender el potencial uso de las bases de datos vectoriales, las
cuales nos permiten reducir la dimensión de la información y explotabilidad,
representando un desafío al momento de rescatar el significado y relevancia del contenido
de este tipo de dato, donde la semántica toma un papel crucial al momento de exponer
una intención comunicativa.

Si bien el centro del proyecto busca justificar y exponer el almacenamiento de vectores
mediante Chroma, el valor agregado del proyecto se centra en el desarrolló una solución
basada en RAG (Retrieval-Augmented Generation) para optimizar la autoatención en
preguntas frecuentes (FAQ).

De esta manera, se combina la recuperación de información relevante desde una base de
datos vectorial con la generación de respuestas naturales mediante un modelo de
lenguaje (LLM), para proporcionar respuestas precisas y contextualizadas, mejorando la
experiencia del usuario en la resolución de dudas comunes.

---
## Generación del dataset
Para la generación del dataset se limitó la generación de datos a la tecnologías de
OpenAI y Gemini, mediante una conexión a su API, a través de dos funciones que se
fundamentan en la generación de mensajes ficticios bajo la utilización de un prompt
similar, más el ajuste de parámetros como la temperatura, el cual facilita la generación de
mensajes más diversos, sin perder la temática principal.

Por otro lado, la base de datos utilizada se distribuye bajo dos tecnologías distintas, para
favorecer la dispersión de la información y tratar de simular la interacción de las personas
con este tipo de información e intención comunicativa.
Finalmente, se destaca que la generación de los mensajes fue realizada en batches de
100 datos por categoría por consulta para openai, y 50 datos por categoría por consulta
para Gemini.

```Python
def generate_dataset(output_file="generated_data.json"):

    # Definir el prompt para GPT-4
    PROMPT = f"""
          Debes generar {n_msj} datos para cada una de las 3 categorías relacionadas a procedimientos bancarios.
          USA las DEFINICIONES de clase como criterios clave para generar los mensajes.

          DEFINICIÓN PARA CADA TIPO DE MENSAJE:

          1.- Banca: Incluir preguntas relacionadas con operaciones bancarias comunes, como consultar saldos, generar cartolas bancarias, cambiar claves, realizar transferencias, abrir cuentas, entre otras.
          2.- Seguros: Crear preguntas relacionadas con seguros ofrecidos por bancos, como seguros para vehículos, seguros de vida o de hogar, seguros de viaje, de vida, de salud, entre otros; incluyendo cómo cotizarlos, contratarlos o cancelarlos.
          3.- Renuncia: Formular preguntas sobre cómo cerrar cuentas bancarias, cancelar servicios o renunciar a productos financieros.

          RECUERDA DEVOLVER {n_msj} MENSAJES PARA CADA CATAEGORÍA Y SOLO EN FORMATO JSON.
          """

    # Completar el texto utilizando el modelo de OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": """Eres un asistente experto en servicios bancarios y de seguros. Tu tarea es generar un conjunto de datos ficticios sobre autogestión bancaria.
                                           Debes devolver un json con la siguiente estructura:
                                            {"Categoría": " ",
                                             "Pregunta": " ",
                                             "Respuesta": " "}"""},
            {"role": "user", "content": PROMPT}
        ]
      )

    # Obtener el mensaje generado
    gen_message = completion.choices[0].message.content.strip()

    print("Respuesta generada por la API:\n", gen_message)  # Depuración

    # Guardar el JSON bruto para inspección
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(gen_message)
        print(f"Archivo JSON generado y guardado en: {output_file}")

    try:
        # Intentar convertir el texto JSON generado en un objeto Python
        data = json.loads(gen_message)
        return data
    except json.JSONDecodeError as e:
        print(f"Error al procesar el JSON generado: {e}")
        return None
```
``` Python
def query_gemini(output_file=None):
    """
    Consulta a la API de Gemini con un prompt y devuelve la respuesta como JSON.
    También guarda la información generada en un archivo JSON si se especifica.

    Args:
        prompt (str): El prompt que se envía al modelo.
        output_file (str): Ruta del archivo para guardar la respuesta generada.

    Returns:
        dict: Respuesta procesada en formato JSON, o None si ocurre un error.
    """

    # Prompt para generar datos ficticios
    prompt = """
    Eres un asistente experto en servicios bancarios y de seguros. Tu tarea es generar un conjunto de datos ficticios sobre autogestión bancaria.
    Debes generar un archivo .json con 50 mensajes para cada una de las 3 categorías relacionadas a procedimientos bancarios.

    USA las DEFINICIONES de clase como criterios clave para generar los mensajes.
    DEFINICIÓN PARA CADA TIPO DE MENSAJE:

    1.- Banca: Incluir preguntas relacionadas con operaciones bancarias comunes, como consultar saldos, generar cartolas bancarias, cambiar claves, realizar transferencias o abrir cuentas.
    2.- Seguros: Crear preguntas relacionadas con seguros ofrecidos por bancos, como seguros para vehículos, seguros de vida o de hogar, incluyendo cómo cotizarlos, contratarlos o cancelarlos.
    3.- Renuncia: Formular preguntas sobre cómo cerrar cuentas bancarias, cancelar servicios o renunciar a productos financieros.

    Debes devolver un json con la siguiente estructura:
      {"Categoría": " ",
        "Pregunta": " ",
        "Respuesta": " "}

    RECUERDA DEVOLVER SOLO EL FORMATO JSON ANTERIOR.
    """

    try:
        # Configurar el modelo de Gemini
        model = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
            generation_config={"temperature": 0.5, # Modificable según variabilidad deseada del dato
                               "response_mime_type": "application/json"}
        )

        # Enviar el prompt a la API
        raw_response = model.generate_content(prompt)

        # Extraer el texto de la respuesta
        response_text = raw_response.text.strip()

        # Verificar si la respuesta contiene múltiples JSON concatenados
        json_objects = []
        for line in response_text.splitlines():
            try:
                json_objects.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error al procesar una línea: {line}")
                print(f"Detalle del error: {e}")

        # Guardar la lista de JSONs en un archivo si se proporciona un nombre de archivo
        if output_file:
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(json_objects, file, indent=2, ensure_ascii=False)
                print(f"Archivo JSON generado y guardado en: {output_file}")

        return json_objects

    except Exception as e:
        print(f"Error al comunicarse con la API de Gemini: {e}")
        return None
```
--- 
# Vectorización de datos
Para la transformación de los textos combinados de cada entrada (Pregunta + Respuesta)
en representaciones numéricas mediante un modelo de embeddings, se utilizó el modelo
text-embedding-ada-002 de OpenAI, que generó vectores de alta dimensionalidad
capaces de capturar las relaciones semánticas en los datos.
Si bien se realizó una evaluación de distintos servicios para la elección del mejor sistema
para vectorizar la información, se optó por este modelo debido a la robustez que
presentan sus servicios.
De esta manera, los embeddings generados con el modelo de OpenAI se almacenan en
un objeto de referencia al origen de los datos ficticios generados, para constatar que su
representación haya sido realizada.

```Python
def get_embedding(text, model="text-embedding-ada-002"):
    """
    Genera un embedding vectorial para un texto utilizando un modelo de embeddings.

    Args:
        text (str): Texto de entrada para el cual se generará el embedding.
        model (str): Nombre del modelo de OpenAI utilizado para generar el embedding.
                     Por defecto, utiliza el modelo "text-embedding-ada-002".

    Returns:
        list: Un vector (embedding) de alta dimensionalidad que representa la semántica del texto.
    """

    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model)

    return response.data[0].embedding

embedding_model = "text-embedding-ada-002"

# Ensure 'documents' is a DataFrame or Series before applying the function
documents = data["Pregunta_respuesta"]  # This assumes 'data' is your DataFrame
documents = documents.to_frame()  # If 'documents' is a Series, convert it to a DataFrame
documents["embedding"] = documents['Pregunta_respuesta'].apply(lambda x: get_embedding(x, model=embedding_model))

# Ensure there are embeddings to concatenate
if len(documents.embedding.values) > 0:
    matrix = np.vstack(documents.embedding.values)
else:
    matrix = np.array([])
```
---
# Almacenamiento y similitud en Chroma
Una vez almacenados los embeddings en ChromaDB, se implementó la funcionalidad de
búsqueda semántica mediante consultas directas al objeto de la colección. La búsqueda
se basa en el método query de ChromaDB, utilizado para realizar una búsqueda de
vecinos más cercanos (nearest neighbors).

Esta técnica identifica los documentos más relevantes al calcular la proximidad entre
vectores, lo que permite recuperar rápidamente el documento o documentos más
relevantes para una consulta específica.

``` Python
def create_chroma_db(documents, collection_name):
    """
    Crea una base de datos vectorial en Chroma a partir de un conjunto de documentos y sus embeddings.

    Args:
        documents (pandas.DataFrame): Un DataFrame que contiene los documentos y sus embeddings precomputados.
                                      Debe incluir las columnas:
                                      - 'Pregunta_respuesta': Texto unificado del documento.
                                      - 'embedding': Embedding asociado al documento.
        collection_name (str): Nombre de la colección que se creará en ChromaDB.

    Returns:
        chromadb.Collection: La colección creada en ChromaDB que contiene los documentos y sus embeddings.

    """
    chroma_client = chromadb.Client()
    embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key)  # Usar la API de OpenAI
    db = chroma_client.create_collection(collection_name, embedding_function=embedding_function)

    # Agregar los documentos y embeddings a la colección
    for i, row in documents.iterrows():
        # Check if the embedding is already a list
        if isinstance(row['embedding'], list):
            embedding = row['embedding']
        else:
            # If not a list, try to convert it using ast.literal_eval
            try:
                embedding = ast.literal_eval(row['embedding'])
            except (SyntaxError, ValueError):
                # If conversion fails, print a warning and skip the row
                print(f"Warning: Could not convert embedding for row {i}. Skipping.")
                continue
        db.add(
            documents=[row["Pregunta_respuesta"]],
            embeddings=[embedding],  # Embeddings precomputados
            ids=[str(i)]
        )

    print(f"Chroma collection '{collection_name}' created with {len(documents)} documents.")
    return db
```
```Python
def query_chroma(query, db, n_results=1):
    """
    Realiza una consulta a la base de datos Chroma para encontrar el texto más relevante.

    Args:
        query (str): Texto de consulta.
        db (chromadb.Collection): Colección de Chroma donde buscar.
        n_results (int): Número de resultados más relevantes a devolver.

    Returns:
        list: Lista de los textos más relevantes encontrados.
    """
    results = db.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if "documents" in results and results["documents"] else []
```
---
# Integración con LLM
La integración con un modelo de lenguaje (LLM) es el paso final del flujo RAG
(Retrieval-Augmented Generation), donde se combina el contexto recuperado desde la
base de datos vectorial con las capacidades generativas del modelo para producir
respuestas completas y personalizadas.

Este proceso inicia cuando el usuario realiza una consulta. La consulta es transformada
en un vector utilizando el mismo modelo de embeddings empleado para procesar los
datos almacenados en la base de datos. Posteriormente, este vector actúa como una
representación semántica de la consulta, permitiendo buscar documentos relevantes
mediante mediciones de similitud.

El sistema utiliza la base de datos vectorial generada mediante ChromaDB para identificar
los documentos más relevantes en función de la consulta del usuario. A través de la
búsqueda semántica basada en la medición de distancia, se recupera un documento que
contiene el contexto necesario para responder a la consulta. Así, este documento
relevante actúa como una pieza central del flujo, proporcionando la información más
precisa y alineada con las necesidades del usuario.

Una vez identificado el documento relevante, el siguiente paso es construir un prompt
dinámico para el modelo de lenguaje (LLM). Este prompt incluye tanto la consulta del
usuario como el contenido del documento recuperado, presentado como un dato de
referencia.

Finalmente, el modelo de lenguaje genera una respuesta basada en el prompt y el
contexto proporcionado. Este enfoque garantiza que las respuestas sean no solo
relevantes, sino también comprensibles y alineadas con las expectativas del usuario.

``` Python
def query_prompt(query, relevant_passage):
    """
    Crea un prompt para el LLM utilizando la consulta del usuario y un documento relevante.

    Args:
        query (str): Consulta realizada por el usuario.
        relevant_passage (str): Documento relevante recuperado de la base de datos.

    Returns:
        str: Prompt contextualizado para el LLM.
    """
    # Limpieza básica del documento relevante
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")

    # Construcción del prompt
    prompt = (f"""
    Eres un asistente útil e informativo que responde preguntas basándote en el texto proporcionado en el pasaje de referencia incluido a continuación.
    Asegúrate de responder de manera completa, siendo comprensivo y proporcionando toda la información de contexto relevante.
    Sin embargo, estás hablando con una audiencia no técnica, así que desglosa los conceptos complicados y utiliza recursos amigable.
    Si la pregunta no es relevante para la respuesta, puedes derivar la respuesta hacia un ejecutivo u oficina de atención.

    PREGUNTA: '{query}'
    PASAJE: '{escaped_passage}'

    RESPUESTA:
    """)
    return prompt
```

``` Python
def generate_response(query, db, model):
    """
    Genera una respuesta personalizada utilizando el LLM y un documento relevante de ChromaDB.

    Args:
        query (str): Consulta realizada por el usuario.
        db: Objeto de la colección ChromaDB.
        model: Objeto del modelo generativo (e.g., Gemini o GPT).

    Returns:
        str: Respuesta generada por el LLM.
    """
    # Recuperar el pasaje relevante desde ChromaDB
    relevant_passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]

    # Crear el prompt contextualizado
    prompt = query_prompt(query, relevant_passage)

    # Generar la respuesta utilizando el modelo generativo
    response = model.generate_content(prompt)

    return response.text
```

