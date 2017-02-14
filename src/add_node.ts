import { Matrix } from 'vectorious';
import { BaseNode } from './node';
import { Value, processMatrices } from './util';

export class AddNode extends BaseNode {
  constructor(nodes: BaseNode[]) {
    super(nodes);
  }

  forward(): void {
    let total: Matrix;

    this.inboundNodes.forEach((n, i) => {
      if (i == 0) {
        total = n.value.map(v => v);
      } else {
        total.add(n.value);
      }
    });

    this.value = total;
  }

  backward(): void {
    this.inboundNodes.forEach(n => {
      this.gradients[n.id] = Matrix.zeros(n.value.shape[0], n.value.shape[1]);
    });

    this.outboundNodes.forEach(n => {
      const gradCost = n.gradients[this.id];

      this.inboundNodes.forEach(n2 => {
        this.gradients[n2.id] = processMatrices('Addition', this.gradients[n2.id], gradCost);
      });
    });
  }
}

export function Add(...nodes: BaseNode[]): AddNode {
  return new AddNode(nodes);
}