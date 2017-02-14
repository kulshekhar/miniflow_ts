import { Matrix } from 'vectorious';
import { BaseNode } from './node';
import { processMatrices } from './util';

export class SigmoidNode extends BaseNode {
  constructor(node: BaseNode) {
    super([node]);
  }

  private sigmoid(x: Matrix) {
    return x.map(v => 1 / (1 + Math.exp(-v)));
  }

  forward(): void {
    this.value = this.sigmoid(this.inboundNodes[0].value);
  }

  backward(): void {
    this.initializeGradients();

    const inputNode = this.inboundNodes[0];
    const sigmoid = this.value;
    const derivative = processMatrices('Multiplication', sigmoid, sigmoid.map(n => 1 - n));

    this.outboundNodes.forEach(n => {
      const gradCost = n.gradients[this.id];

      // console.log(gradCost);

      const product = processMatrices('Multiplication', derivative, gradCost);

      this.gradients[inputNode.id] = processMatrices('Addition',
        this.gradients[inputNode.id], product);
    });
  }
}

export function Sigmoid(node: BaseNode): SigmoidNode {
  return new SigmoidNode(node);
}