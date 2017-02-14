import { Matrix } from 'vectorious';
import { BaseNode } from './node';
import { Value, processMatrices } from './util';

export class MSENode extends BaseNode {
  diff: Matrix;
  m: number;

  constructor(y: BaseNode, a: BaseNode) {
    super([y, a]);
  }

  forward() {
    const y = this.inboundNodes[0].value;
    const a = this.inboundNodes[1].value;

    this.m = y.shape[0];
    this.diff = processMatrices('Subtraction', y, a);

    let total = 0;
    this.diff.each(n => { total += n * n; });

    const mean = total / (this.diff.shape[0] * this.diff.shape[1]);

    this.value = new Matrix([[mean]]);
  }

  backward(): void {
    this.gradients[this.inboundNodes[0].id] = processMatrices('Multiplication', this.diff, new Matrix([[2 / this.m]]));

    this.gradients[this.inboundNodes[1].id] = processMatrices('Multiplication', this.diff, new Matrix([[-2 / this.m]]));
  }
}

export function MSE(y: BaseNode, a: BaseNode): MSENode {
  return new MSENode(y, a);
}