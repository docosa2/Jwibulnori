using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace TensorTests
{
    [TestClass]
    public class TensorTests
    {
        [TestMethod] // Tensor 생성 시 Shape과 Data 배열 크기 확인
        public void Tensor_Initialization_CreatesCorrectShapeAndSize()
        {
            var shape = new int[] { 2, 3 };
            var tensor = new Tensor(shape);

            CollectionAssert.AreEqual(shape, tensor.Shape);
            Assert.AreEqual(6, tensor.Data.Length); // 2 * 3
        }

        [TestMethod] // 인덱스를 이용한 데이터 접근/수정이 올바르게 작동하는지 확인
        public void Tensor_Indexing_SetsAndGetsCorrectValues()
        {
            var tensor = new Tensor(new int[] { 2, 2 });

            tensor[0, 0] = 1.5f;
            tensor[0, 1] = 2.5f;
            tensor[1, 0] = 3.5f;
            tensor[1, 1] = 4.5f;

            Assert.AreEqual(1.5f, tensor[0, 0]);
            Assert.AreEqual(2.5f, tensor[0, 1]);
            Assert.AreEqual(3.5f, tensor[1, 0]);
            Assert.AreEqual(4.5f, tensor[1, 1]);
        }

        [TestMethod] // 내부 인덱싱 로직 (GetOffset)이 예상대로 작동하는지 확인
        public void Tensor_GetOffset_ReturnsCorrectOffset()
        {
            var tensor = new Tensor(new int[] { 3, 4 });
            var offset = tensor.GetOffset(new int[] { 2, 1 }); // 2 * 4 + 1 = 9

            Assert.AreEqual(9, offset);
        }

        [TestMethod] // 잘못된 차원 수의 인덱싱 시 예외 발생 여부 확인
        [ExpectedException(typeof(ArgumentException))]
        public void Tensor_Indexing_ThrowsOnInvalidIndexLength()
        {
            var tensor = new Tensor(new int[] { 2, 2 });

            // This should throw ArgumentException
            var _ = tensor[0, 1, 2];
        }
    }

    [TestClass]
    public class TensorExtendedTests
{
    [TestMethod]
    public void Tensor_Reshape_MaintainsDataOrder()
    {
        var tensor = new Tensor(new int[] { 2, 2 });
        tensor[0, 0] = 1;
        tensor[0, 1] = 2;
        tensor[1, 0] = 3;
        tensor[1, 1] = 4;

        var reshaped = tensor.Reshape(new int[] { 4 });

        Assert.AreEqual(1, reshaped[0]);
        Assert.AreEqual(2, reshaped[1]);
        Assert.AreEqual(3, reshaped[2]);
        Assert.AreEqual(4, reshaped[3]);
    }

    [TestMethod]
    public void Tensor_Clone_CreatesIndependentCopy()
    {
        var tensor = new Tensor(new int[] { 2 });
        tensor[0] = 42;

        var clone = tensor.Clone();
        clone[0] = 99;

        Assert.AreEqual(42, tensor[0]);
        Assert.AreEqual(99, clone[0]);
    }

    [TestMethod]
    public void Tensor_CopyTo_CopiesDataCorrectly()
    {
        var source = new Tensor(new int[] { 3 });
        source[0] = 10; source[1] = 20; source[2] = 30;

        var target = new Tensor(new int[] { 3 });
        source.CopyTo(target);

        CollectionAssert.AreEqual(source.Data, target.Data);
    }

    [TestMethod]
    public void Tensor_3DInitialization_SetsAndGetsCorrectly()
    {
        var tensor = new Tensor(new int[] { 2, 2, 2 });

        tensor[0, 0, 0] = 1.0f;
        tensor[0, 1, 1] = 2.0f;
        tensor[1, 0, 1] = 3.0f;
        tensor[1, 1, 0] = 4.0f;

        Assert.AreEqual(1.0f, tensor[0, 0, 0]);
        Assert.AreEqual(2.0f, tensor[0, 1, 1]);
        Assert.AreEqual(3.0f, tensor[1, 0, 1]);
        Assert.AreEqual(4.0f, tensor[1, 1, 0]);
    }
}

}
