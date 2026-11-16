"""Score a whole folder concurrently from async code."""
import asyncio
import sys
from pathlib import Path

from anyjev import Judge, presets


async def main(folder: str) -> None:
    task = presets.structured_vs_prose()
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    async with Judge(concurrency=32) as judge:
        results = await judge.aclassify_many(task, images, on_result=print)
    structured = [r.input_id for r in results if r.ok and r.is_positive]
    print(f"\n{len(structured)}/{len(results)} crops need full extraction")


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "."))
