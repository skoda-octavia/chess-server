import { Component, OnInit, ViewChild } from '@angular/core';
import {NgxChessBoardService} from 'ngx-chess-board';
import {NgxChessBoardView} from 'ngx-chess-board';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  @ViewChild('board', { static: false })
  board!: NgxChessBoardView;

  turn = true;
  isGameFinished = false;
  endgameMessage = '';

  constructor(private ngxChessBoardService: NgxChessBoardService) {}

  ngOnInit(): void {}

  ngAfterViewInit() {
    this.board.reset();
  }

  switchPlayerTurn() {
    this.turn = !this.turn;
  }

  MoveCompleted(event: any) {
    console.log(event)
    if (event.checkmate && this.turn) {
      this.endgameMessage = 'white won!';
      this.isGameFinished = true;
      console.warn(this.endgameMessage)
      return;
    }

    if (event.stalement) {
      this.endgameMessage = 'draw!';
      this.isGameFinished = true;
      console.warn(this.endgameMessage)
      return;
    }
    console.log(this.board.getFEN());
    this.switchPlayerTurn();
  }

  onResetGame() {
    this.board.reset();
    this.turn = true;
    this.endgameMessage = '';
    this.isGameFinished = false;
  }
}