# RUST_LOG used for setting the logging level (only in debug runs)
RUST_LOG=trace

.PHONY: build

build:
	@ if [ -f "target/release/lux-runtime.out" ]; then rm target/release/lux-runtime.out; fi
	cargo build --release
	@ mv target/release/lux-runtime target/release/lux-runtime.out

.PHONY: build-dev

build-dev:
	@ if [ -f "target/debug/lux-runtime.out" ]; then rm target/debug/lux-runtime.out; fi
	cargo build
	@ mv target/debug/lux-runtime target/debug/lux-runtime.out

.PHONY: run

run: build-dev
	RUST_LOG=${RUST_LOG} luxai-s2 target/debug/lux-runtime.out target/debug/lux-runtime.out --out=replay.json -v 3
