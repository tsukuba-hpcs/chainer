import argparse


def main(log):
    map_async = []
    prefetch_from_backend = []
    prefetch_multiprocess_iterator_cached_id_queue_get = []
    prefetch_multiprocess_iterator_cached_id_queue_put = []

    with open(log, "r") as f:
        for l in f:
            l = l.strip()
            if l.startswith("_prefetch_from_backend:"):
                prefetch_from_backend.append(
                    float(l.replace("_prefetch_from_backend: ", ""))
                )

            if l.startswith("_prefetch_multiprocess_iterator_cached_id_queue.get"):
                prefetch_multiprocess_iterator_cached_id_queue_get.append(
                    float(
                        l.replace(
                            "_prefetch_multiprocess_iterator_cached_id_queue.get: ", ""
                        )
                    )
                )

            if l.startswith("map_async: "):
                map_async.append(float(l.replace("map_async: ", "")))

            if l.startswith("_prefetch_multiprocess_iterator_cached_id_queue.put"):
                prefetch_multiprocess_iterator_cached_id_queue_put.append(
                    float(
                        l.replace("_prefetch_multiprocess_iterator_cached_id_queue.put: ", "")
                    )
                )    

    print(
        f"{sum(prefetch_from_backend)}," + 
        f"{sum(prefetch_multiprocess_iterator_cached_id_queue_put)}," +
        f"{sum(prefetch_multiprocess_iterator_cached_id_queue_get)}," +
        f"{sum(map_async)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", "-l", type=str)
    args = parser.parse_args()

    main(args.log)

