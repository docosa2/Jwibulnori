using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace TensorTests
{
    [TestClass]
    public class TensorTests
    {
        [TestMethod]
        public void Tensor_Initialization_CreatesCorrectShapeAndSize()
        {
            var shape = new int[] { 2, 3 };
            var tensor = new Tensor<float>(shape);

            CollectionAssert.AreEqual(shape, tensor.Shape);
            Assert.AreEqual(6, tensor.Length); // 2 * 3
        }

        [TestMethod]
        public void Tensor_Indexing_SetsAndGetsCorrectValues()
        {
            var tensor = new Tensor<float>(new int[] { 2, 2 });

            tensor[0, 0] = 1.5f;
            tensor[0, 1] = 2.5f;
            tensor[1, 0] = 3.5f;
            tensor[1, 1] = 4.5f;

            Assert.AreEqual(1.5f, tensor[0, 0]);
            Assert.AreEqual(2.5f, tensor[0, 1]);
            Assert.AreEqual(3.5f, tensor[1, 0]);
            Assert.AreEqual(4.5f, tensor[1, 1]);
        }

        [TestMethod]
        public void Tensor_GetOffset_ReturnsCorrectOffset()
        {
            var tensor = new Tensor<float>(new int[] { 3, 4 });
            var offset = tensor.GetOffset(new int[] { 2, 1 }); // 2 * 4 + 1 = 9

            Assert.AreEqual(9, offset);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Tensor_Indexing_ThrowsOnInvalidIndexLength()
        {
            var tensor = new Tensor<float>(new int[] { 2, 2 });

            // 잘못된 인덱스 길이로 접근 시 예외 발생
            var _ = tensor[0, 1, 2];
        }

        [TestMethod]
        public void Tensor_Reshape_MaintainsDataOrder()
        {
            var tensor = new Tensor<int>(new int[] { 2, 2 });
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
            var tensor = new Tensor<int>(new int[] { 2 });
            tensor[0] = 42;

            var clone = tensor.Clone();
            clone[0] = 99;

            Assert.AreEqual(42, tensor[0]);
            Assert.AreEqual(99, clone[0]);
        }

        [TestMethod]
        public void Tensor_CopyTo_CopiesDataCorrectly()
        {
            var source = new Tensor<int>(new int[] { 3 });
            source[0] = 10; source[1] = 20; source[2] = 30;

            var target = new Tensor<int>(new int[] { 3 });
            source.CopyTo(target);

            for (int i = 0; i < source.Length; i++)
            {
                Assert.AreEqual(source[i], target[i]);
            }
        }

        [TestMethod]
        public void Tensor_3DInitialization_SetsAndGetsCorrectly()
        {
            var tensor = new Tensor<float>(new int[] { 2, 2, 2 });

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
