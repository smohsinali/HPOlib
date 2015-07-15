#!/usr/bin/env python

from argparse import ArgumentParser
import os

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def main():
    description = "Output's HPOlib-run commands for test functions"
    parser = ArgumentParser(description=description)

    # General Options
    parser.add_argument("-b", "--benchmark", dest="benchmarkDir",
                        default="../benchmarks/branin", help="specify benchmark name")
    parser.add_argument("-opt", "--optDir", dest="optimizerDir",
                        default="../optimizers", help="path to optimizers directory")
    parser.add_argument("-o", "--outputFile", dest="Output",
                        default="HPOlib-run commands", help="outputFile")
    parser.add_argument("-start-seed", "--start_seed", default=1000,
                        type=int, help="seed start value")
    parser.add_argument("-step-seed", "--step_seed", default=1000,
                        type=int, help="seed step value")
    parser.add_argument("-stop-seed", "--stop_seed", default=10000,
                        type=int, help="seed stop value")

    args, unknown = parser.parse_known_args()

    if os.path.isabs(args.optimizerDir):
        optimizersDir = args.optimizerDir
    else:
        optimizersDir = os.path.abspath(args.optimizerDir)

    if os.path.isabs(args.benchmarkDir):
        benchmarkDir = args.benchmarkDir
    else:
        benchmarkDir = os.path.abspath(args.benchmarkDir)

    output = args.Output
    startSeed = args.start_seed
    stepSeed = args.step_seed
    stopSeed = args.stop_seed

    optimizers = {}
    opts = os.walk(optimizersDir).next()[1]
    for opt in opts:
        optVersions = []
        optimizerDir = os.path.join(optimizersDir, opt)
        insideOpt = os.walk(optimizerDir).next()[2]
        for s in filter(lambda x: ".cfg" in x, insideOpt): optVersions.append(s[0:-11])
        optimizers[opt] = optVersions

    outputfile = open(output, 'w')
    dirs = os.walk(benchmarkDir, followlinks=True).next()[1]

    for seed in my_range(startSeed, stopSeed, stepSeed):
        for optimizer in optimizers:
            for optVersion in optimizers[optimizer]:
                if optVersion in dirs:
                    path = os.path.join(optimizersDir, optimizer, optVersion)
                    outputfile.write("HPOlib-run --cwd " + benchmarkDir + " -o " + path + " -s " + str(seed) + "\n")


if __name__ == '__main__':
    main()
