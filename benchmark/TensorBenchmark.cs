using BenchmarkDotNet.Attributes;

[MemoryDiagnoser]
public class TensorBenchmark
{
    private Tensor<float> tensorA;
    private Tensor<float> tensorB;
    private Tensor<float> result;

    [GlobalSetup]
    public void Setup()
    {
        tensorA = new Tensor<float>(new int[] { 1000 });
        tensorB = new Tensor<float>(new int[] { 1000 });
        result = new Tensor<float>(new int[] { 1000 });

        for (int i = 0; i < 1000; i++)
        {
            tensorA.Data[i] = i;
            tensorB.Data[i] = 2 * i;
        }
    }

    [Benchmark]
    public void ElementwiseAdd()
    {
        for (int i = 0; i < tensorA.Data.Length; i++)
        {
            result.Data[i] = tensorA.Data[i] + tensorB.Data[i];
        }
    }

    [Benchmark]
    public void CloneTensor()
    {
        var clone = tensorA.Clone();
    }

    [Benchmark]
    public void ReshapeTensor()
    {
        var reshaped = tensorA.Reshape(new int[] { 500, 2 });
    }
}

