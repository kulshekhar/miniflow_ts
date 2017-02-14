import { Matrix } from 'vectorious';
import { BaseNode } from './node';
import { InputNode } from './input_node';
import * as fs from 'fs';

export function topologicalSort(feedDict: { [key: string]: NodeAndValue }): BaseNode[] {
  const inputNodes = Object.keys(feedDict).map((k) => feedDict[k].node);
  const G: { [key: string]: NodeGraph } = {};
  let nodes = inputNodes.slice(0);

  while (nodes.length > 0) {
    const n = nodes.splice(0, 1)[0];
    if (!G[n.id]) {
      G[n.id] = new NodeGraph();
    }
    n.outboundNodes.forEach(m => {
      if (!G[m.id]) {
        G[m.id] = new NodeGraph();
      }

      G[n.id].out.add(m);
      G[m.id].in.add(n);

      nodes.push(m);
    });
  }

  const L: BaseNode[] = [];
  let S = new Set(inputNodes);

  while (S.size > 0) {
    const n = S.values().next().value;
    S.delete(n);

    if (n instanceof InputNode) {
      n.value =
        Object.keys(feedDict).
          filter(k => feedDict[k].node.id == n.id).
          map(k => feedDict[k].value)[0];
    }

    L.push(n);

    n.outboundNodes.forEach(m => {
      G[n.id].out.delete(m);
      G[m.id].in.delete(n);
      if (G[m.id].in.size == 0) {
        S.add(m);
      }
    });
  }

  return L;
}

export function forwardAndBackward(outputNode: BaseNode, sortedNodes: BaseNode[]): number | Matrix {

  sortedNodes.forEach(n => {
    n.forward();
  });

  sortedNodes.reverse().forEach(n => {
    n.backward();
  });

  if (outputNode.value.shape[0] == 1 && outputNode.value.shape[1] == 1) {
    return outputNode.value.get(0, 0);
  }

  return outputNode.value;
}

export function sgdUpdate(trainables: BaseNode[], learningRate: number = 0.01) {
  trainables.forEach(n => {
    n.value = processMatrices('Subtraction',
      n.value, Matrix.scale(n.gradients[n.id], learningRate));
  });
}

export class NodeAndValue {
  node: BaseNode;
  value: Matrix;
}

export function Value(
  n: BaseNode,
  v: number | number[] | number[][] | Matrix,
  transpose: boolean = false): NodeAndValue {

  if (Array.isArray(v)) {
    if (v.length == 0) return { node: n, value: new Matrix([]) };

    if (Array.isArray(v[0])) return { node: n, value: new Matrix(v as number[][]) };


    if (typeof v[0] == 'number') {
      if (transpose) {
        return { node: n, value: new Matrix([v as number[]]).T };
      }

      return { node: n, value: new Matrix([v as number[]]) };
    }

  } else if (typeof v === 'number') {

    return { node: n, value: new Matrix([[v]]) };

  } else if (v instanceof Matrix) {

    return { node: n, value: v };
  }

  throw 'Invalid value type';
}


class NodeGraph {
  readonly in = new Set();
  readonly out = new Set();
}

export function randomString(length: number = 16, chars: string = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'): string {
  var result = '';
  for (var i = length; i > 0; --i) result += chars[Math.floor(Math.random() * chars.length)];
  return result;
}

export type MatrixOperation = 'Addition' | 'Subtraction' | 'Multiplication' | 'Division';

export function processMatrices(operation: MatrixOperation, x: Matrix, y: Matrix): Matrix {

  const xRows = x.shape[0];
  const xCols = x.shape[1];
  const yRows = y.shape[0];
  const yCols = y.shape[1];

  const isSameShape = xRows == yRows && xCols == yCols;
  const isDottable = xCols == yRows;

  const process = function (a: number, b: number): number {
    switch (operation) {
      case 'Addition':
        return a + b;
      case 'Subtraction':
        return a - b;
      case 'Multiplication':
        return a * b;
      case 'Division':
        return a / b;
    }
    throw 'Invalid operation';
  }

  if (xRows == yRows && yCols == 1) {
    const newMatrix = x.map(v => v);
    for (let i = 0; i < xRows; i++) {
      for (let j = 0; j < xCols; j++) {
        newMatrix.set(i, j, process(x.get(i, j), y.get(i, 0)));
      }
    }
    return newMatrix;
  }

  if (xRows == yRows && xCols == 1) {
    const newMatrix = y.map(v => v);
    for (let i = 0; i < xRows; i++) {
      for (let j = 0; j < yCols; j++) {
        newMatrix.set(i, j, process(y.get(i, j), x.get(i, 0)));
      }
    }
    return newMatrix;
  }

  if (xCols == yCols && yRows == 1) {
    const newMatrix = x.map(v => v);
    for (let i = 0; i < xRows; i++) {
      for (let j = 0; j < xCols; j++) {
        newMatrix.set(i, j, process(x.get(i, j), y.get(0, j)));
      }
    }
    return newMatrix;
  }

  if (xCols == yCols && xRows == 1) {
    const newMatrix = y.map(v => v);
    for (let i = 0; i < yRows; i++) {
      for (let j = 0; j < xCols; j++) {
        newMatrix.set(i, j, process(y.get(i, j), x.get(0, j)));
      }
    }
    return newMatrix;
  }

  if (yRows == 1 && yCols == 1) {
    const newMatrix = x.map(v => v);
    for (let i = 0; i < xRows; i++) {
      for (let j = 0; j < xCols; j++) {
        newMatrix.set(i, j, process(x.get(i, j), y.get(0, 0)));
      }
    }
    return newMatrix;
  }

  if (xRows == 1 && xCols == 1) {
    const newMatrix = y.map(v => v);
    for (let i = 0; i < yRows; i++) {
      for (let j = 0; j < yCols; j++) {
        newMatrix.set(i, j, process(y.get(i, j), x.get(0, 0)));
      }
    }
    return newMatrix;
  }

  // if (operation == 'Multiplication' && !isSameShape && isDottable) {
  //   return Matrix.multiply(x, y);
  // }

  try {
    switch (operation) {
      case 'Addition':
        return Matrix.add(x, y);
      case 'Subtraction':
        return Matrix.subtract(x, y);
      case 'Multiplication': {
        if (isSameShape) return Matrix.product(x, y);
        if (isDottable) return Matrix.multiply(x, y);
      }
    }

    throw 'Invalid operation attempted on the matrices';
  } catch (e) {
    console.log(`Operation: ${operation}
Dottable: ${isDottable}
Same Shape: ${isSameShape}

X (${x.shape}): 
${x} 

Y (${y.shape}): 
${y}`);

    throw e;
  }
}

export function addMatrixColumns(m: Matrix): Matrix {
  const mRows = m.shape[0];
  const mCols = m.shape[1];
  const result = Matrix.zeros(1, mCols);

  for (var i = 0; i < mCols; i++) {
    for (var j = 0; j < mRows; j++) {
      result.set(0, i, result.get(0, i) + m.get(j, i));
    }
  }

  return result;
}

export function getBostonData(): [number[][], number[][]] {
  const content = fs.readFileSync('./data/housing.data', 'utf-8');

  const lines = content.split('\n').filter(l => l.trim().length > 0);

  const table = lines.map(l =>
    l.split(' ').
      filter(w => w != '').
      map(w => w.indexOf('.') >= 0 ? parseFloat(w) : parseInt(w)));

  const input = table.map(row => row.slice(0, 13));
  const output = table.map(row => row.slice(13));

  return [input, output];
}

export function normalize2DArray(a: number[][]): number[][] {
  const m = new Matrix(a);
  const aT = m.T.toArray();

  const [rows, cols] = m.shape;

  const mean: number[] = aT.map(row => row.reduce((p, c) => p + c, 0) / row.length);

  const std: number[] = aT.map((row, i) => {
    const avg = mean[i];

    const avgSquaredDiffs = row.reduce((p, c) => {
      const diff = c - avg;
      return p + diff * diff;
    }, 0) / row.length;

    return Math.sqrt(avgSquaredDiffs);
  });

  const meanMatrix = new Matrix([mean]);
  const stdMatrix = new Matrix([std]);

  const diff = processMatrices('Subtraction', m, meanMatrix);
  const normalized = processMatrices('Division', diff, stdMatrix);

  return normalized.toArray();
}

export function resample(x: number[][], y: number[][], samples: number = 10): [number[][], number[][]] {

  const resultX: number[][] = [];
  const resultY: number[][] = [];

  const usedIndices = new Set();

  while (resultX.length < samples) {
    const index = Math.floor(Math.random() * x.length);
    if (usedIndices.has(index)) continue;
    usedIndices.add(index);
    resultX.push(x[index]);
    resultY.push(y[index]);
  }

  return [resultX, resultY];
}