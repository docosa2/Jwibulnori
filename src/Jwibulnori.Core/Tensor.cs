// 2025-04-17 by Kwak, M.D.

public class Tensor
{
    public float[] Data;
    public int[] Shape;
    public int[] Strides;

    public Tensor(int[] shape)
    {
        Shape = shape;
        Data = new float[Shape.Aggregate((a, b) => a * b)];
        Strides = ComputeStrides(Shape);
    }

    public Tensor(float[] data, int[] shape)
    {
        if (data.Length != shape.Aggregate((a, b) => a * b))
            throw new ArgumentException("Data length does not match shape.");
        Data = (float[])data.Clone();
        Shape = (int[])shape.Clone();
        Strides = ComputeStrides(Shape);
    }

    private int[] ComputeStrides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        int acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    public int GetOffset(int[] indices)
    {
        if (indices.Length != Shape.Length)
            throw new ArgumentException("Dimension mismatch.");
        int offset = 0;
        for (int i = 0; i < indices.Length; i++)
            offset += indices[i] * Strides[i];
        return offset;
    }

    public float this[params int[] indices]
    {
        get => Data[GetOffset(indices)];
        set => Data[GetOffset(indices)] = value;
    }

    public Tensor Reshape(int[] newShape)
    {
        if (newShape.Aggregate((a, b) => a * b) != Data.Length)
            throw new ArgumentException("Total size must remain unchanged.");
        return new Tensor(Data, newShape);
    }

    public Tensor Clone()
    {
        return new Tensor(Data, Shape);
    }

    public void CopyTo(Tensor target)
    {
        if (target.Data.Length != Data.Length)
            throw new ArgumentException("Target tensor size mismatch.");
        Array.Copy(this.Data, target.Data, Data.Length);
    }
}