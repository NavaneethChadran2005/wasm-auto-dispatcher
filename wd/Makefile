# Requires clang (>= 15) with the wasm32 target and wasm-ld (package "lld").
CC      = clang
OUT     := public/wasm
COMMON  := --target=wasm32 -O3 -nostdlib -ffreestanding -fno-builtin -Wall
LDCOMMON:= -Wl,--no-entry -Wl,--export=matmul_rows -Wl,--export=saxpy_range \
           -Wl,--import-memory -Wl,--stack-first -Wl,-z,stack-size=4096
# 1 GiB max (16384 pages). JS grows memory as needed.
MAXMEM  := 1073741824

all: $(OUT)/kernels_scalar.wasm $(OUT)/kernels_simd.wasm $(OUT)/kernels_simd_mt.wasm

$(OUT)/kernels_scalar.wasm: src/kernels.c
	$(CC) $(COMMON) $< -o $@ $(LDCOMMON) -Wl,--initial-memory=1048576 -Wl,--max-memory=$(MAXMEM)

$(OUT)/kernels_simd.wasm: src/kernels.c
	$(CC) $(COMMON) -msimd128 $< -o $@ $(LDCOMMON) -Wl,--initial-memory=1048576 -Wl,--max-memory=$(MAXMEM)

$(OUT)/kernels_simd_mt.wasm: src/kernels.c
	$(CC) $(COMMON) -msimd128 -matomics -mbulk-memory $< -o $@ $(LDCOMMON) \
	  -Wl,--shared-memory -Wl,--initial-memory=1048576 -Wl,--max-memory=$(MAXMEM)

clean:
	rm -f $(OUT)/*.wasm
