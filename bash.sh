#!/bin/bash

# Folders
INPUT_DIR="./25_nov_2025"
OUTPUT_VIDEOS_DIR="./results"
OUTPUT_CSV_DIR="./annotations"

mkdir -p "$OUTPUT_VIDEOS_DIR"
mkdir -p "$OUTPUT_CSV_DIR"

echo "Processing 1.mp4 ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/1.mp4" \
    --output_video "$OUTPUT_VIDEOS_DIR/1.mp4" \
    --output_csv "$OUTPUT_CSV_DIR/1.csv" \
    --model yolo11l.pt \
    --conf 0.01

echo "Processing 2.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/2.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/2.mov" \
    --output_csv "$OUTPUT_CSV_DIR/2.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 3.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/3.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/3.mov" \
    --output_csv "$OUTPUT_CSV_DIR/3.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 4.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/4.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/4.mov" \
    --output_csv "$OUTPUT_CSV_DIR/4.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 5.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/5.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/5.mov" \
    --output_csv "$OUTPUT_CSV_DIR/5.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 6.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/6.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/6.mov" \
    --output_csv "$OUTPUT_CSV_DIR/6.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 7.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/7.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/7.mov" \
    --output_csv "$OUTPUT_CSV_DIR/7.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 8.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/8.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/8.mov" \
    --output_csv "$OUTPUT_CSV_DIR/8.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 9.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/9.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/9.mov" \
    --output_csv "$OUTPUT_CSV_DIR/9.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 10.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/10.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/10.mov" \
    --output_csv "$OUTPUT_CSV_DIR/10.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 11.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/11.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/11.mov" \
    --output_csv "$OUTPUT_CSV_DIR/11.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 12.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/12.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/12.mov" \
    --output_csv "$OUTPUT_CSV_DIR/12.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 13.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/13.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/13.mov" \
    --output_csv "$OUTPUT_CSV_DIR/13.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 14.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/14.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/14.mov" \
    --output_csv "$OUTPUT_CSV_DIR/14.csv" \
    --model yolo11l.pt \
    --conf 0.1

echo "Processing 15.mov ..."
python3 ./code/main.py \
    --input "$INPUT_DIR/15.mov" \
    --output_video "$OUTPUT_VIDEOS_DIR/15.mov" \
    --output_csv "$OUTPUT_CSV_DIR/15.csv" \
    --model yolo11l.pt \
    --conf 0.1
