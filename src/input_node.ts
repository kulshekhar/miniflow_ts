import { Matrix } from 'vectorious';
import { BaseNode } from './node';
import { processMatrices } from './util';

export class InputNode extends BaseNode {

  forward(value?: number | number[] | number[][]): void {
    // if (value) {
    //   if (typeof value === 'number') {
    //     this.value = new Matrix([[value]]);
    //   } else if (Array.isArray(value)) {
    //     if (value.length == 0) {
    //       this.value = new Matrix([]);
    //     } else {
    //       this.value = (Array.isArray(value[0]))
    //         ? new Matrix(<number[][]>value)
    //         : new Matrix([<number[]>value]);
    //     }
    //   } else {
    //     throw 'Invalid type for the input node';
    //   }
    // }
  }

  backward() {
    this.gradients[this.id] = Matrix.zeros(1, 1);

    this.outboundNodes.forEach(n => {
      const gradCost = n.gradients[this.id];
      this.gradients[this.id] = processMatrices('Addition', this.gradients[this.id], gradCost);
    });
  }
}

export function Input(): InputNode {
  return new InputNode();
};