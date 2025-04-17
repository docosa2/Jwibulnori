// 2025-04-17 by Kwak, M.D.
using System;
using System.Buffers;
using System.Linq;

public class Tensor<T>
{
    public int[] Shape { get; }
    public int[] Strides { get; }
    public T[] Data { get; private set;}
    public int Rank => Shape.Length;
    public int Length => Data.Length;

    public Tensor(params int[] shape)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape must be a non-empty array.");

        Shape = (int[])shape.Clone();
        Strides = ComputeStrides(Shape);
        Data = new T[ComputeLength(Shape)];
    }

    private static int[] ComputeStrides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int ComputeLength(int[] shape)
    {
        int length = 1;
        foreach (var dim in shape)
            length *= dim;
        return length;
    }

    public int GetOffset(int[] indices)
    {
        if (indices.Length != Rank)
            throw new ArgumentException($"Expected {Rank} indices, but got {indices.Length}.");

        int offset = 0;
        for (int i = 0; i < Rank; i++)
        {
            if (indices[i] < 0 || indices[i] >= Shape[i])
                throw new ArgumentOutOfRangeException($"Index {i} is out of bounds.");
            offset += indices[i] * Strides[i];
        }
        return offset;
    }

    public T this[params int[] indices]
    {
        get => Data[GetOffset(indices)];
        set => Data[GetOffset(indices)] = value;
    }

    public Tensor<T> Reshape(int[] newShape)
    {
        if (ComputeLength(newShape) != Length)
            throw new ArgumentException("Total size of new shape must be unchanged.");

        var reshaped = new Tensor<T>(newShape)
        {
            Data = Data // Sharing the same data array
        };
        return reshaped;
    }

    public Tensor<T> Clone()
    {
        var clone = new Tensor<T>(Shape);
        Array.Copy(Data, clone.Data, Length);
        return clone;
    }

    public void CopyTo(Tensor<T> destination)
    {
        if (destination == null)
            throw new ArgumentNullException(nameof(destination));
        if (destination.Length != Length)
            throw new ArgumentException("Destination tensor must have the same length.");
        Array.Copy(Data, destination.Data, Length);
    }
}
