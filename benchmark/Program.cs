#pragma warning disable CA1050
using BenchmarkDotNet.Running;

public class Program
{
    public static void Main(string[] args)
    {
        BenchmarkRunner.Run<TensorBenchmark>();
    }
}