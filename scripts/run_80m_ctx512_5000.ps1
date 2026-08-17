$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

python src/train.py --model baseline --out-dir runs/baseline80m_512_5000 --d-model 744 --layers 12 --heads 12 --rank 64 --batch-size 1 --grad-accum 1 --seq-len 512 --steps 5000 --logical-epochs 10 --lr 0.0002 --min-lr 0.00002 --warmup 250 --eval-interval 500 --eval-batches 5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/train.py --model procedural --out-dir runs/procedural80m_512_5000 --d-model 744 --layers 12 --heads 12 --rank 64 --batch-size 1 --grad-accum 1 --seq-len 512 --steps 5000 --logical-epochs 10 --lr 0.0012 --min-lr 0.00012 --warmup 250 --eval-interval 500 --eval-batches 5
exit $LASTEXITCODE
