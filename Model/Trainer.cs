using System;
using System.Collections.Generic;
using System.Net.Configuration;

namespace NeuralNet.Model
{
    // Train a neural net
    public class Trainer
    {
        SimpleNeuralNet neuralNet;
        DataSet dataSet;
        int miniBatchSize = 0;
        float learningRate;
        Action<Result> resultAction;
        Random rand;

        List<int> shuffled;

        public struct Result
        {
            public string message;
            public int maxEpochs;
            public int curEpoch;
            public int batchIndex;
            public int numBatches;
            public int trainingSize;
            public int trainingProcessed;
            public float trainingError;
            public int testSize;
            public int testProcessed;
            public float testError;
            public int elapsedMs;
            public string boundsName;
            public float boundsValue;
            public override string ToString()
            {
                return $"Epoch {curEpoch}/{maxEpochs}, batch {batchIndex}/{numBatches}" +
                       $", train & test error ({trainingError:F3},{testError:F3})" +
                       $", Elapsed ms {elapsedMs}, bounds {boundsName}{boundsValue}";
            }
        };

        Result state;

        void Shuffle(Random rand, List<int> items)
        {
            var len = items.Count;
            for (var i = 0; i < len; ++i)
                items[i] = items[i + rand.Next(len - i)];
        }
        public void Train(
            // the net to train
            SimpleNeuralNet neuralNet,
            // the data set (training and validation)
            DataSet dataSet,
            // source of randomness
            Random rand,
            // where to put notifications
            Action<Result> resultAction = null,
            // number of epochs to do (complete passes through data)
            int epochs = 100,
            // number to do per mini batch
            int miniBatchSize = 100, 
            // learning rate
            float learningRate = 0.1f
        )
        {
            this.neuralNet = neuralNet;
            this.dataSet = dataSet;
            this.rand = rand;
            this.state.maxEpochs = epochs;
            this.state.trainingSize = dataSet.TrainingSet.Count;
            this.state.testSize = dataSet.TestSet.Count;
            this.miniBatchSize = miniBatchSize;
            this.learningRate = learningRate;
            this.resultAction = resultAction;
            this.state.message = "";

            state.curEpoch = state.trainingProcessed = state.testProcessed = 0;

            shuffled = new List<int>();
            for (var i = 0; i < state.trainingSize; ++i)
                shuffled.Add(i);

            int startMs = Environment.TickCount, endMs = 0;
            state.numBatches = (state.trainingSize + miniBatchSize - 1) / miniBatchSize;

            for (var epoch = 0; epoch < state.maxEpochs; epoch++)
            {
                Shuffle(rand, shuffled);

                state.curEpoch = epoch+1;
                state.trainingProcessed = 0;
                state.testProcessed = 0;

                // do training via SGD for one epoch
                var totalTrainingPassed = 0;
                for (var minibatch = 0; minibatch < state.numBatches; ++minibatch)
                {
                    state.batchIndex = minibatch + 1;
                    var batchStart = minibatch * miniBatchSize;
                    var batchSize = Math.Min(miniBatchSize, state.trainingSize - batchStart);
                    totalTrainingPassed += ProcessBatch(
                        i => dataSet.TrainingSet[shuffled[i + batchStart]],
                        batchSize,
                        true
                        );
                    state.trainingProcessed += batchSize;
                    state.trainingError = (float)totalTrainingPassed / state.trainingProcessed;
                    endMs = Environment.TickCount;
                    state.elapsedMs = endMs - startMs;
                    (state.boundsName, state.boundsValue) = neuralNet.Bounds();
                    resultAction(state);
                }

                // check against test data
                var totalTestPassed = ProcessBatch(
                    i => dataSet.TestSet[i],
                    state.testSize,
                    false
                    );
                state.testError = (float)totalTestPassed / state.testSize;
                state.trainingProcessed += this.miniBatchSize;
                endMs = Environment.TickCount;
                state.elapsedMs = endMs - startMs;
                (state.boundsName, state.boundsValue) = neuralNet.Bounds();
                resultAction(state);

            }
        }

        public static int MaxIndex(Vector vector)
        {
            // find max index
            var maxIndex = 0;
            var maxValue = vector[0];
            for (var i = 1; i < vector.Size; ++i)
            {
                if (maxValue < vector[i])
                {
                    maxValue = vector[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        // return true if computed passed compared to desired
        bool Passed(Vector computed, Vector desired)
        {
            var computedIndex = MaxIndex(computed);
            var desiredIndex = MaxIndex(desired);
            return computedIndex == desiredIndex;
        }

        // process data, return number correct
        int ProcessBatch(Func<int,DataPoint> getPoint, int batchSize, bool training)
        {

            // run through dataset, total avg error
            var totalPassed = 0;
            if (training)
                neuralNet.ZeroDeltas();
            for (var i = 0; i < batchSize; ++i)
            {
                var point = dataSet.TrainingSet[shuffled[i]];
                var output = neuralNet.FeedForward(point.input);
                //var (nameT, boundsT) = neuralNet.Bounds(); // watch errors - todo
                var errorVector = output - point.output;

                totalPassed += Passed(output, point.output) ? 1 : 0;

                if (training)
                    neuralNet.BackpropagateToDeltas(errorVector);
                //(nameT, boundsT) = neuralNet.Bounds(); // watch errors - todo
                //neuralNet.Check();
                //Console.WriteLine($"Train {i+1}/{trainingSize} -> {nameT} {boundsT}");
            }
            if (training)
                neuralNet.ModifyWeights(learningRate, batchSize);
            return totalPassed;
        }
    }
}
