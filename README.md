# Lambda Gemini API (Python)

Función AWS Lambda escrita en Python que recibe un `prompt` y genera imágenes usando la API de Google Gemini. Incluye infraestructura basada en AWS SAM y un pipeline de despliegue automático con GitHub Actions.

## Requisitos previos

- Python 3.12+
- AWS CLI configurado con permisos para Lambda, API Gateway, CloudFormation, S3 y SSM Parameter Store.
- SAM CLI
- Una cuenta con acceso a la API de Google Gemini y una API key activa.

## Variables y secretos necesarios

1. **AWS Systems Manager Parameter Store**: crea un parámetro seguro (SecureString) que contenga tu API key de Gemini.
   ```bash
   aws ssm put-parameter \
     --name /prod/gemini/api-key \
     --type SecureString \
     --value "TU_API_KEY"
   ```

2. **GitHub Secrets** para el pipeline (`Settings` > `Secrets and variables` > `Actions`):
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` (por ejemplo `us-east-1`)
   - `SSM_PARAMETER_NAME` (por ejemplo `/prod/gemini/api-key`)

3. **GitHub Variables** opcionales (`Settings` > `Secrets and variables` > `Actions` > `Variables`):
   - `AWS_STACK_NAME` (default `lambda-gemini-api`)
   - `LAMBDA_FUNCTION_NAME` (default `lambda-gemini-api`)

## Estructura principal

- `src/app.py`: handler de Lambda que invoca Gemini y devuelve las imágenes en base64.
- `template.yaml`: plantilla SAM para desplegar la función y exponer un endpoint HTTP.
- `.github/workflows/deploy.yaml`: workflow que ejecuta pruebas y despliega automáticamente con SAM.
- `requirements.txt`: dependencias de producción empaquetadas junto con la función.

## Ejecución local de pruebas

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

> **Nota:** Este entorno no cuenta con acceso a Internet, por lo que las instalaciones de dependencias pueden fallar localmente. Ejecuta los comandos anteriores en un entorno con red para preparar el proyecto antes del primer despliegue.

## Despliegue manual

```bash
sam build --cached --parallel
sam deploy \
  --stack-name lambda-gemini-api \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides \
      GoogleApiKeyParameterName=/prod/gemini/api-key \
      LambdaFunctionName=lambda-gemini-api
```

## Payload de ejemplo

```json
{
  "prompt": "Un paisaje futurista al atardecer, estilo ilustración digital",
  "mimeType": "image/png"
}
```

La respuesta devuelve un arreglo `images` con objetos que incluyen `mimeType` y `data` (contenido base64 listo para guardarse como archivo o renderizarse en un `<img>` con un data URI).

## Flujo del pipeline

1. Al hacer push a `main` (o manual con `workflow_dispatch`) se dispara el workflow.
2. Se instala Python 3.12, las dependencias y se ejecutan las pruebas con `pytest`.
3. Se configura el acceso a AWS con los secretos definidos.
4. SAM compila y despliega la plantilla, actualizando o creando la función y el API Gateway.

Tras el despliegue, el output `ApiEndpoint` (disponible en CloudFormation) brinda la URL para consumir la Lambda vía HTTP.
