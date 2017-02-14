import { Matrix, Vector } from 'vectorious';
import { Input, InputNode } from './input_node';
import { BaseNode } from './node';
import { Add, AddNode } from './add_node';
import { Multiply } from './multiply_node';
import { Linear } from './linear_node';
import { MSE } from './mse_node';
import { Sigmoid } from './sigmoid_node';
import { Value, forwardAndBackward, topologicalSort, getBostonData, normalize2DArray, resample, sgdUpdate, processMatrices } from './util';


[
  sgd,
  backprop,
  mse,
  sigmoid,
  linear,
  multiply,
  add,

].forEach((f, i) => {
  i == 0 && console.log('-------------');
  f();
  console.log('-------------');
});

function sgd() {
  const [X_, y_] = getBostonData();
  const nX_ = normalize2DArray(X_);
  const featureCount = nX_[0].length;
  const nHidden = 10;
  const W1_ = Matrix.random(featureCount, nHidden);
  const b1_ = Matrix.zeros(1, nHidden);
  const W2_ = Matrix.random(nHidden, 1);
  const b2_ = Matrix.zeros(1, 1);

  const [X, y] = [Input(), Input()];
  const [W1, b1] = [Input(), Input()];
  const [W2, b2] = [Input(), Input()];

  const l1 = Linear(X, W1, b1);
  const s1 = Sigmoid(l1);
  const l2 = Linear(s1, W2, b2);
  const cost = MSE(y, l2);

  const feedDict = {
    X: Value(X, X_),
    y: Value(y, y_),
    W1: Value(W1, W1_),
    b1: Value(b1, b1_),
    W2: Value(W2, W2_),
    b2: Value(b2, b2_),
  };

  const epochs = 10;
  const m = X_.length;
  const batchSize = 11;
  const stepsPerEpoch = Math.floor(m / batchSize);

  const graph = topologicalSort(feedDict);
  const trainables = [W1, b1, W2, b2];

  console.log(`Total number of examples: ${m}`);

  for (let i = 0; i < epochs; i++) {
    let loss = 0;
    for (let j = 0; j < stepsPerEpoch; j++) {
      const [xBatch, yBatch] = resample(X_, y_, batchSize);

      X.value = new Matrix(xBatch);
      y.value = new Matrix(yBatch);

      forwardAndBackward(cost, graph);
      sgdUpdate(trainables);

      loss += graph[graph.length - 1].value.get(0, 0);
    }
    console.log(`Epoch: ${i + 1}, Loss: ${loss / stepsPerEpoch}`);
  }
}

function backprop() {
  const [X, W, b, y] = [Input(), Input(), Input(), Input()];
  const f = Linear(X, W, b);
  const a = Sigmoid(f);
  const cost = MSE(y, a);

  const feedDict = {
    X: Value(X, [[-1., -2.], [-1, -2]]),
    W: Value(W, [[2.], [3.]]),
    b: Value(b, [-3.]),
    y: Value(y, [1, 2], true),
  }

  const graph = topologicalSort(feedDict);
  const output = forwardAndBackward(cost, graph);
  const gradients = [X, y, W, b].map(n => n.gradients[n.id]);

  console.log(`Backprop: ${gradients}`);
}

function mse() {
  const [y, a] = [Input(), Input()];
  const cost = MSE(y, a);

  const feedDict = {
    y: Value(y, [1, 2, 3], true),
    a: Value(a, [4.5, 5, 10], true),
  };

  const graph = topologicalSort(feedDict);
  const output = forwardAndBackward(cost, graph);

  console.log(`MSE: ${output}`);
}

function sigmoid() {
  const [X, W, b] = [Input(), Input(), Input()];
  const f = Linear(X, W, b);
  const g = Sigmoid(f);

  const feedDict = {
    X: Value(X, [[-1., -2.], [-1, -2]]),
    W: Value(W, [[2., -3], [2., -3]]),
    b: Value(b, [-3., -5]),
  }

  const graph = topologicalSort(feedDict);
  const output = forwardAndBackward(g, graph);

  console.log(`Sigmoid: ${output}`);
}

function linear() {
  const [X, W, b] = [Input(), Input(), Input()];
  const f = Linear(X, W, b);

  const feedDict = {
    X: Value(X, [6, 14, 3]),
    W: Value(W, [0.5, 0.25, 1.4], true),
    b: Value(b, 2),
  }

  const graph = topologicalSort(feedDict);
  const output = forwardAndBackward(f, graph);

  console.log(`Linear: ${output}`);
}

function multiply() {
  const [x, y, z] = [Input(), Input(), Input()];
  const f = Multiply(x, y, z);

  const feedDict = {
    x: Value(x, 10),
    y: Value(y, 5),
    z: Value(z, 3),
  };

  const graph = topologicalSort(feedDict);
  const output = forwardAndBackward(f, graph);

  console.log(`${x} * ${y} * ${z} = ${output}`);
}

function add() {
  const [x, y, z] = [Input(), Input(), Input()];
  const f = Add(x, y, z);

  const feedDict = {
    x: Value(x, 10),
    y: Value(y, 5),
    z: Value(z, 3),
  };

  const graph = topologicalSort(feedDict);
  const output = forwardAndBackward(f, graph);

  console.log(`${x} + ${y} + ${z} = ${output}`);
}