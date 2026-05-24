//
//  Item.swift
//  Smart Receipts
//
//  Created by Coring on 5/24/26.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
