| Dataset | Images / boxes | Classes | Image size (px) | Defect-free images | Audit findings | Split (train/val/test) |
|---|---|---|---|---|---|---|
| NEU-DET (steel strip) | 1800 / 4189 | 6 | 200 × 200 | 0 | 2 byte-identical images (1 pair); 188 boxes of co-occurring classes | stratified by class; duplicate pair kept together; 1079/361/360 |
| GC10-DET (steel sheet) | 2306 / 3542 | 10 | 2048 × 1000 | 0 | 956 acquisition sequences (≤ 133 frames); 12 images in two class folders; 26 without annotation; 73 XML/folder class disagreements; 1 unknown label | grouped by sequence: 1416/396/468 (0 shared sequences); image-random: 1368/456/456 (306 test images share a sequence with training) |
| Magnetic Tile (ferrite tile) | 1344 / 443 | 5 | 103–632 × 222–403 | 952 | same tile under several exposures: 557 tile groups (67 of six images); boxes from mask components; 6 defective images without a component ≥ 9 px | grouped by tile, stratified: 835/243/266 (86 defective test images) |
