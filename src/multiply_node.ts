import { Matrix } from 'vectorious';
import { BaseNode } from './node';
import { processMatrices } from './util';

export class MultiplyNode extends BaseNode {
  constructor(nodes: BaseNode[]) {
    super(nodes);
  }

  forward(): void {
    let total: Matrix;

    this.inboundNodes.forEach((n, i) => {
      if (i == 0) {
        total = n.value.map(v => v);
      } else {
        total.product(n.value);
      }
    });

    this.value = total;
  }

  backward(): void {
    this.inboundNodes.forEach(n => {
      this.gradients[n.id] = Matrix.zeros(n.value.shape[0], n.value.shape[1]);
    });

    this.outboundNodes.forEach((n, i) => {
      const gradCost = n.gradients[this.id];

      this.inboundNodes.forEach((n2, i2) => {
        let product = n2.value.map(_ => 1);

        this.inboundNodes
          .filter(n3 => n3.id != n2.id)
          .forEach(n3 => {
            product = processMatrices('Multiplication', product, n3.value);
          });

        const nodeGrad = Matrix.multiply(gradCost, product);

        this.gradients[n2.id] = processMatrices('Addition', this.gradients[n2.id], nodeGrad);
      });
    });
  }
}

export function Multiply(...nodes: BaseNode[]): MultiplyNode {
  return new MultiplyNode(nodes);
}