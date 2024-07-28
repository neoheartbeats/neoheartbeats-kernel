from qdrant_client import QdrantClient, AsyncQdrantClient, models
import numpy as np
import asyncio


async def main():
    client = AsyncQdrantClient(url="http://192.168.4.8:6333")
    await client.create_collection(
        collection_name="my_collection",
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    await client.upsert(
        collection_name="my_collection",
        points=[
            models.PointStruct(
                id=i,
                vector=np.random.rand(10).tolist(),
            )
            for i in range(100)
        ],
    )

    res = await client.search(
        collection_name="my_collection",
        query_vector=np.random.rand(10).tolist(),  # type: ignore
        limit=10,
    )

    print(res)


asyncio.run(main())
