.PHONY: audit test manifest verify-manifest

audit:
	python scripts/release_audit.py --root . --strict

test:
	python -m unittest discover -s tests -p 'test_release_*.py'

manifest:
	python scripts/create_release_manifest.py --root . --output RELEASE_MANIFEST.json

verify-manifest:
	python scripts/create_release_manifest.py --root . --verify RELEASE_MANIFEST.json
