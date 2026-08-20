# Tech Challenge B - Pipeline de IA

Este projeto executa um pipeline completo de Machine Learning para classificar
se um caso deve ser marcado como perigoso a partir de animal e sintomas.

## Execucao local

```bash
python techchallengeB.py --data data.csv --output outputs_techchallenge_b
```

## Principais artefatos gerados

- `reports/relatorio_tecnico.md`
- `reports/relatorio_tecnico.pdf`
- `tables/*.csv` e `tables/*.json`
- `figures/*.png`
- `models/best_model.pkl`

## Docker

Copie `techchallengeB.py`, `data.csv` e este Dockerfile para a mesma pasta:

```bash
docker build -t techchallenge-b .
docker run --rm -v "%cd%/outputs:/app/outputs" techchallenge-b
```
