import argparse


def main(log):
    elapsed_time = 0
    with open(log, "r") as f:
        for l in f:
            l = l.strip()
            if l.startswith("--train") or len(l) == 0:
                continue
            else:
                elapsed_time = float(l)

    print(elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", "-l", type=str)
    args = parser.parse_args()

    main(args.log)

