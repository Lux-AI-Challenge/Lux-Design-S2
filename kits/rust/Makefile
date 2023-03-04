.PHONY: build

build:
	@ if [ -f "target/release/lux-runtime.out" ]; then rm target/release/lux-runtime.out; fi
	cargo build --release
	@ mv target/release/lux-runtime target/release/lux-runtime.out

.PHONY: run

run: build
	RUST_LOG=trace luxai-s2 target/release/lux-runtime.out target/release/lux-runtime.out --out=replay.json -v 3
