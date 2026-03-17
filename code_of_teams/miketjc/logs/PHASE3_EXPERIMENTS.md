# Phase 3 Inference Experiments - Tracking

## Current Status (2026-03-11 15:36 CDT)

### ✓ Active Experiments (Running)

#### 1. Stride-64 + 8-way Ensemble
- **Account**: eric
- **GPU**: 0 (100% utilization)
- **Model**: Phase 2j
- **Checkpoint**: raim_mef_phase2j/snapshot/net_final.pth
- **Configuration**:
  - stride: 64 (87.5% overlap)
  - ensemble: 8-way geometric
  - tile_size: 512
  - jpeg_quality: 95
  - in_channels: 15
- **Progress**: 1/100 (started generating)
- **ETA**: ~5 hours (20:30 CDT)
- **Expected Score**: 57.5-57.8
- **Output**: `/u/jtu9/scratch/NTR/contests/raim_mef/inference_results/val_phase3_s64_ens/`

#### 2. Stride-96 + 8-way Ensemble  
- **Account**: cici
- **GPU**: 1 (99% utilization)
- **Model**: Phase 2j
- **Configuration**:
  - stride: 96 (81.25% overlap)
  - ensemble: 8-way geometric
  - tile_size: 512
  - jpeg_quality: 95
  - in_channels: 15
- **Progress**: 3/100
- **ETA**: ~1.6 hours (16:36 CDT)
- **Expected Score**: 57.4-57.6
- **Output**: `/u/jtu9/scratch/NTR/contests/raim_mef/inference_results/val_phase3_s96_ens/`

### ✓ Baseline Reference
- **Phase 3j** (stride-128 + 8-way): **57.1426** (submission ID 614702)

### Automatic Systems

#### Monitoring
- Real-time progress: `/tmp/realtime_monitor.log` (updates every minute)
- Inference status: `/tmp/inference_status.log` (checks every 5 minutes)
- Completion notification: `/tmp/wait_and_notify.log` (checks every 5 minutes)

#### ZIP Preparation
- Automatic ZIP creation: `/tmp/prepare_submissions.py`
- Status log: `/tmp/prepare_submissions.log`
- Triggers when all 100 JPGs generated

### Submission Files (Ready Once Inference Completes)
- stride-96: `/u/jtu9/scratch/NTR/contests/raim_mef/raim_mef_phase3_s96_ens.zip`
- stride-64: `/u/jtu9/scratch/NTR/contests/raim_mef/raim_mef_phase3_s64_ens.zip`

### Next Steps
1. Wait for stride-96 completion (~16:36 CDT) → Create ZIP → Submit via eric account
2. Wait for stride-64 completion (~20:30 CDT) → Create ZIP → Submit via cici account
3. Monitor scores and identify best performer
4. Queue subsequent experiments if needed:
   - stride-128 + 16-way ensemble (miketjc0316)
   - Phase 2j + Phase 2i model ensemble (yaoxin)

## Key Metrics
- **Stride impact**: stride-64 ~87.5% more tiles per image than stride-96
- **Processing rate stride-96**: ~2 images/hour (based on early samples)
- **Processing rate stride-64**: ~0.3 images/hour (based on early samples)
- **Both GPUs**: Running at sustained 100% utilization

## Test Data
- Location: `/u/jtu9/scratch/dataset/NTR/Raim_Mef/testdata-phase3/testdata_phase3/`
- Format: 100 scenes, each with 5 multi-exposure images
- Image size: 2040×1528 pixels
- Total: 500 images (100 × 5)

## Model Checkpoint
- Path: `/u/jtu9/scratch/NTR/contests/raim_mef/raim_mef_phase2j/snapshot/net_final.pth`
- Size: 544 MB
- Architecture: TimeDiffiT_ResNet_color_128
- Params: 142M
- Training: Phase 2j (λp=0.8, L1+SSIM+LPIPS, 300 epochs from Phase 2c base)

## Score Progression
| Phase | Config | Score |
|-------|--------|-------|
| 2j | stride-128+ens | 56.6913 |
| 3j | stride-128+ens | **57.1426** |
| 3-s96 (est) | stride-96+ens | 57.4-57.6 |
| 3-s64 (est) | stride-64+ens | 57.5-57.8 |

